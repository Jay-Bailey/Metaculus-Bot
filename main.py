import asyncio
import datetime
import json
import os
import requests
import re
import sys
from tqdm.asyncio import tqdm
from urllib.parse import urlencode

from anthropic import AsyncAnthropic, InternalServerError, RateLimitError
import backoff
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}_forecaster.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


# Create a logger object
logger = logging.getLogger(__name__)

METACULUS_TOKEN = os.environ.get('METACULUS_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PERPLEXITY_API_KEY = os.environ.get('PERPLEXITY_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
logger.info("Environment variables loaded: " + str(METACULUS_TOKEN is not None) + ", " + str(PERPLEXITY_API_KEY is not None) + ", " + str(ANTHROPIC_API_KEY is not None))
MODEL = 'claude-3-5-sonnet-20240620'

PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.
The interviewer is also a professional forecaster, with a strong track record of
accurate forecasts of the future. They will ask you a question, and your task is
to provide the most accurate forecast you can. To do this, you evaluate past data
and trends carefully, make use of comparison classes of similar events, take into
account base rates about how past events unfolded, and outline the best reasons
for and against any particular outcome. You know that great forecasters don't
just forecast according to the "vibe" of the question and the considerations.
Instead, they think about the question in a structured way, recording their
reasoning as they go, and they always consider multiple perspectives that
usually give different conclusions, which they reason about together.
You can't know the future, and the interviewer knows that, so you do not need
to hedge your uncertainty, you are simply trying to give the most accurate numbers
that will be evaluated when the events later unfold.

Your interview question is:
{title}

Your research assistant says:
{summary_report}

background:
{background}

fine_print:
{fine_print}

resolution_criteria:
{resolution_criteria}

Today is {today}.

You write your rationale and give your final answer as: "Probability: ZZ%", 0-100
"""

logger.info("Metaculus token loaded: " + str(METACULUS_TOKEN is not None))
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api2"
WARMUP_TOURNAMENT_ID = 3294
SUBMIT_PREDICTION = True
logger.info("Prediction submission enabled: " + str(SUBMIT_PREDICTION))

def find_number_before_percent(s):
    # Use a regular expression to find all numbers followed by a '%'
    matches = re.findall(r'(\d+)%', s)
    if matches:
        # Return the last number found before a '%'
        return int(matches[-1])
    else:
        # Return None if no number found
        return None

def post_question_comment(question_id, comment_text):
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/",
        json={
            "comment_text": comment_text,
            "submit_type": "N",
            "include_latest_prediction": True,
            "question": question_id,
        },
        **AUTH_HEADERS,
    )
    response.raise_for_status()

def post_question_prediction(question_id, prediction_percentage):
    """
    Post a prediction value (between 1 and 100) on the question.
    """
    url = f"{API_BASE_URL}/questions/{question_id}/predict/"
    response = requests.post(
        url,
        json={"prediction": float(prediction_percentage) / 100},
        **AUTH_HEADERS,
    )
    response.raise_for_status()


def get_question_details(question_id):
    """
    Get all details about a specific question.
    """
    url = f"{API_BASE_URL}/questions/{question_id}/"
    response = requests.get(
        url,
        **AUTH_HEADERS,
    )
    response.raise_for_status()
    return json.loads(response.content)

def list_questions(tournament_id=3349, offset=0, count=10, get_answered_questions=False):
    """
    List (all details) {count} questions from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "project": tournament_id,
        "status": "open",
        "type": "forecast",
        "include_description": "true",
    }
    if not get_answered_questions:
        url_qparams["not_guessed_by"] = 190772

    url = f"{API_BASE_URL}/questions/"
    logging.info(f"Requesting {url}{urlencode(url_qparams)}")
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    response.raise_for_status()
    data = json.loads(response.content)
    return data

@backoff.on_exception(backoff.expo,
                      requests.exceptions.HTTPError,
                      max_tries=8,  # Adjust as needed
                      factor=2,     # Exponential factor
                      jitter=backoff.full_jitter)
def call_perplexity(query):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "content-type": "application/json",
    }
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {
                "role": "system",
                "content": """
You are an assistant to a superforecaster.
The superforecaster will give you a question they intend to forecast on.
To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
You do not produce forecasts yourself.
""",
            },
            {"role": "user", "content": query},
        ],
    }
    response = requests.post(url=url, json=payload, headers=headers)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return content


@backoff.on_exception(backoff.expo,
                      (RateLimitError, InternalServerError),
                      max_tries=8,  # Adjust as needed
                      factor=2,     # Exponential factor
                      jitter=backoff.full_jitter)
async def call_claude(content: str) -> str:
  async_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

  message = await async_client.messages.create(
      model="claude-3-5-sonnet-20240620",
      temperature=0.7,
      max_tokens=4096,
      system="You are a world-class forecaster.",
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": content
                  }
              ]
          }
      ]
  )

  return message.content[0].text, message.usage

async def get_prediction(question_details, model):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    summary_report = call_perplexity(question_details["title"])

    content = PROMPT_TEMPLATE.format(
                title=question_details["title"],
                summary_report=summary_report,
                today=today,
                background=question_details["description"],
                fine_print=question_details["question"]["fine_print"],
                resolution_criteria=question_details["question"]["resolution_criteria"],
            )

    response_text, usage = await call_claude(content)

    # Regular expression to find the number following 'Probability: '
    probability_match = find_number_before_percent(response_text)

    # Extract the number if a match is found
    probability = None
    if probability_match:
        probability = int(probability_match) # int(match.group(1))
        logger.info(f"The extracted probability is: {probability}%")
        probability = min(max(probability, 1), 99) # To prevent extreme forecasts

    return probability, summary_report, response_text, usage

SUMMARY_PROMPT_PREFIX = """Attached is a series of reasons other AI models gave for a prediction."""
SUMMARY_PROMPT_SUFFIX = """Please return a brief summary of the most common considerations, and whether this made the prediction higher or lower."""

def get_usage(model, result):
  if model.startswith('claude'):
    return {'input': result[-1].input_tokens, 'output': result[-1].output_tokens}
  else:
    raise NotImplementedError("Only Claude supported for this atm")

COST_DICT = {"claude-3-5-sonnet-20240620": {"input": 3.0/1000000, "output": 15.0/1000000}}

def get_cost(model, token_dict):
  if model not in COST_DICT:
    raise NotImplementedError("Only Claude supported for this atm")
  return token_dict['input'] * COST_DICT[model]['input'] + token_dict['output'] * COST_DICT[model]['output']

async def process_agent(prediction_fn, question_detail, model, pbar):
    result = await prediction_fn(question_detail, model)
    pbar.update(1)
    return result

async def ensemble_async(model, prediction_fn, question_ids, num_agents=32):
    question_details = [get_question_details(question_id) for question_id in question_ids]
    total_cost = 0
    final_predictions = []
    summaries = []
    total_iterations = len(question_details) * num_agents
    with tqdm(total=total_iterations) as pbar:
        for i, question_detail in enumerate(question_details):
            logger.info(f"Question {i+1} of {len(question_details)}: {question_detail['id']} - {question_detail['title']}")
            tasks = []
            for _ in range(num_agents):
                task = asyncio.create_task(process_agent(prediction_fn, question_detail, model, pbar))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            predictions, usages, response_texts = [result[0] for result in results], [get_usage(model, result) for result in results], [result[2] for result in results]
            final_predictions.append(predictions)
            costs = [get_cost(model, usage) for usage in usages]
            total_cost += sum(costs)
            summary, _ = await call_claude(SUMMARY_PROMPT_PREFIX + '\n'.join(response_texts) + SUMMARY_PROMPT_SUFFIX)
            summaries.append(summary)

            logger.info(predictions)
            aggregated_prediction = round(sum(filter(None, predictions)) / len(list(filter(None, predictions))), 0)
            logger.info(aggregated_prediction)

            if aggregated_prediction is not None and SUBMIT_PREDICTION:
                comment = f"This prediction was made by averaging an ensemble of {num_agents} agents. A summary of their most common considerations follows.\n\n{summaries[i]}"
                logger.info(comment)
                post_question_prediction(question_ids[i], aggregated_prediction)
                post_question_comment(question_ids[i], comment)
                logger.info(f"Prediction submitted for question {question_ids[i]}")

    logger.info(f"Total cost was ${round(total_cost, 2)}")

DEBUG_MODE = False
SUBMIT_PREDICTION = not DEBUG_MODE

# TODO: Incorporate TOURNAMENT_ID, API_BASE_URL, and USER_ID as env variables into the code.

def main():
    data = list_questions(tournament_id=32506, count=2 if DEBUG_MODE else 99, get_answered_questions=DEBUG_MODE)
    ids = [question["id"] for question in data["results"]]
    logger.info(f"Questions found: {ids}")
    results = asyncio.run(ensemble_async(MODEL, get_prediction, ids, num_agents=2 if DEBUG_MODE else 32))
    logger.info(results)

if __name__ == "__main__":
    main()