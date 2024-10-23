import asyncio
import aiohttp
import datetime
import json
import os
import requests
import re
import sys
from tqdm.asyncio import tqdm
from urllib.parse import urlencode
import itertools

from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError, InternalServerError as OpenAIInternalServerError
from anthropic import AsyncAnthropic, InternalServerError as AnthropicInternalServerError, RateLimitError as AnthropicRateLimitError
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
ASK_NEWS_CLIENT_ID = os.environ.get('ASK_NEWS_CLIENT_ID')
ASK_NEWS_CLIENT_SECRET = os.environ.get('ASK_NEWS_CLIENT_SECRET')
API_BASE_URL = "https://www.metaculus.com/api2"
METACULUS_PROXY = False

# Log the names of the env variables too.
logger.info("Environment variables loaded: METACULUS_TOKEN " + str(METACULUS_TOKEN is not None) + ", PERPLEXITY_API_KEY " + str(PERPLEXITY_API_KEY is not None) + ", ANTHROPIC_API_KEY " + str(ANTHROPIC_API_KEY is not None) + ", OPENAI_API_KEY " + str(OPENAI_API_KEY is not None) + ", ASK_NEWS_CLIENT_ID " + str(ASK_NEWS_CLIENT_ID is not None) + ", ASK_NEWS_CLIENT_SECRET " + str(ASK_NEWS_CLIENT_SECRET is not None))
logger.info("METACULUS_PROXY: " + str(METACULUS_PROXY))

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

SUPERFORECASTING_TEMPLATE = """
You are an advanced AI system which has been finetuned to provide calibrated probabilistic forecasts under uncertainty, with your performance evaluated according to the Brier score. When forecasting, do not treat 1% (1:99 odds) and 5% (1:19) as similarly "small" probabilities, or 90% (9:1) and 99% (99:1) as similarly "high" probabilities. As the odds show, they are markedly different, so output your probabilities accordingly.

Question:
{title}

Today's date: {today}
Your pretraining knowledge cutoff: October 2023

We have retrieved the following information for this question:
<background>{summary_report}</background>

Recall the question you are forecasting:
{title}

Question description:
{background}

Relevant fine-print:
{fine_print}

Resolution criteria:
{resolution_criteria}

Instructions:

1. Compress key factual information from the sources, as well as useful background information which may not be in the sources, into a list of core factual points to reference. Aim for information which is specific, relevant, and covers the core considerations you'll use to make your forecast. For this step, do not draw any conclusions about how a fact will influence your answer or forecast. Place this section of your response in <facts></facts> tags.

2. Provide a few reasons why the answer might be no. Rate the strength of each reason on a scale of 1-10. Use <no></no> tags.

3. Provide a few reasons why the answer might be yes. Rate the strength of each reason on a scale of 1-10. Use <yes></yes> tags.

4. Aggregate your considerations. Do not summarize or repeat previous points; instead, investigate how the competing factors and mechanisms interact and weigh against each other. Factorize your thinking across (exhaustive, mutually exclusive) cases if and only if it would be beneficial to your reasoning. We have detected that you overestimate world conflict, drama, violence, and crises due to news' negativity bias, which doesn't necessarily represent overall trends or base rates. Similarly, we also have detected you overestimate dramatic, shocking, or emotionally charged news due to news' sensationalism bias. Therefore adjust for news' negativity bias and sensationalism bias by considering reasons to why your provided sources might be biased or exaggerated. Think like a superforecaster. Use <thinking></thinking> tags for this section of your response.

5. Output an initial probability (prediction) as a single number between 0 and 1 given steps 1-4. Use <tentative></tentative> tags.

6. Reflect on your answer, performing sanity checks and mentioning any additional knowledge or background information which may be relevant. Check for over/underconfidence, improper treatment of conjunctive or disjunctive conditions (only if applicable), and other forecasting biases when reviewing your reasoning. Consider priors/base rates, and the extent to which case-specific information justifies the deviation between your tentative forecast and the prior. Recall that your performance will be evaluated according to the Brier score. Be precise with tail probabilities. Leverage your intuitions, but never change your forecast for the sake of modesty or balance alone. Finally, aggregate all of your previous reasoning and highlight key factors that inform your final forecast. Use <thinking></thinking> tags for this portion of your response.

7. Output your final prediction as: "Probability: ZZ%", 0-100."""

logger.info("Metaculus token loaded: " + str(METACULUS_TOKEN is not None))
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api2"
WARMUP_TOURNAMENT_ID = 3294
SUBMIT_PREDICTION = True
logger.info("Prediction submission enabled: " + str(SUBMIT_PREDICTION))

def find_number_before_percent(s):
    # Use a regular expression to find all numbers (including decimals) prefaced by Probability: and followed by a '%'
    matches = re.findall(r'(\d+(?:\.\d+)?)%', s)
    if matches:
        return float(matches[-1])
    else:
        logger.info(f"No number found in string: {s}")
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
                      (AnthropicRateLimitError, AnthropicInternalServerError, aiohttp.ContentTypeError),
                      max_tries=8,  # Adjust as needed
                      factor=2,     # Exponential factor
                      jitter=backoff.full_jitter)
async def call_claude(content: str) -> str:
    if METACULUS_PROXY:
        url = "https://www.metaculus.com/proxy/anthropic/v1/messages/"

        headers = {
            "Authorization": f"Token {METACULUS_TOKEN}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": "claude-3-5-sonnet-20240620",
            "temperature": 0.7,
            "max_tokens": 4096,
            "system": "You are a world-class forecaster.",
            "messages": [
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
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response_data = await response.json()
                return response_data['content'][0]['text'], response_data['usage']
            
    else:
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

@backoff.on_exception(backoff.expo,
                      (OpenAIRateLimitError, OpenAIInternalServerError),
                      max_tries=8,  # Adjust as needed
                      factor=2,     # Exponential factor
                      jitter=backoff.full_jitter)
async def call_gpt(content: str):
    async_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL or "https://api.openai.com/v1")
    message = await async_client.chat.completions.create(
        model="o1-preview",
        messages = [
            #{"role": "system", "content": "You are a world-class forecaster."},
            {"role": "user", "content": content},
        ]
    )
    return message.choices[0].message.content, message.usage

@backoff.on_exception(backoff.expo,
                      requests.exceptions.HTTPError,
                      max_tries=8,  # Adjust as needed
                      factor=2,     # Exponential factor
                      jitter=backoff.full_jitter)
async def call_ask_news(query, summariser_fn=call_claude):
    ask = AskNewsSDK(client_id=ASK_NEWS_CLIENT_ID, client_secret=ASK_NEWS_CLIENT_SECRET)
    categories = ["All", "Business", "Crime", "Politics", "Science", "Sports", "Technology", "Military", "Health", "Entertainment", "Finance", "Culture", "Climate", "Environment", "World"]
    category_response, _ = await summariser_fn(f"Given the question, what is the most relevant category or categories of news to search for? Question: {query}. Respond with a Python list of categories. If you are unsure, respond with ['All']. This is for an API, so you must pick ONLY from the following categories: {', '.join(categories)}")
    
    try:
        category_list = eval(category_response)
        if not isinstance(category_list, list):
            raise ValueError("Category list must be a list, got: " + str(category_response))
    except Exception as e:
        logger.error(f"Failed to parse categories: {category_response}. Got error: {e}")
        category_list = ["All"]
    
    if not all(category in categories for category in category_list):
        logger.error(f"Invalid categories: {category_list}. Must be a subset of: {categories}. Using ['All'] instead.")
        category_list = ["All"]

    print(category_list)
    graph = ask.news.search_news(query=query, strategy='latest news', diversify_sources=True, method='nl', categories=category_list, return_type="string")
    
    summary, _ = await summariser_fn("You are an assistant to a superforecaster. The superforecaster will give you a question they intend to forecast on. To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information. You do not produce forecasts yourself. The relevant news you have found is: " + graph.as_string)

    if DEBUG_MODE:
        logger.info(f"News summary: {summary}")
    return summary

async def get_prediction(question_details, summary_report, model_fn=call_claude, prompt_template=PROMPT_TEMPLATE):
    """Expected formats:
    
    news_fn(question_title: str) -> str
    model_fn(content: str) -> str
    prompt_template: Contains title, summary, today, background, fine_print, resolution_criteria
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    summary_report = summary_report or await call_ask_news(question_details["title"])
  
    content = prompt_template.format(
        title=question_details["title"],
        summary_report=summary_report,
        today=today,
        background=question_details["question"]["description"],
        fine_print=question_details["question"]["fine_print"],
        resolution_criteria=question_details["question"]["resolution_criteria"],
    )

    response_text, usage = await model_fn(content)

    # Regular expression to find the number following 'Probability: '
    probability_match = find_number_before_percent(response_text)
    logger.info(f"Probability match: {probability_match}")

    if probability_match:
        probability = int(probability_match) # int(match.group(1))
        logger.info(f"The extracted probability is: {probability}%")
        probability = min(max(probability, 1), 99) # To prevent extreme forecasts
    else:
        probability = None

    return probability, summary_report, response_text, usage

SUMMARY_PROMPT_PREFIX = """Attached is a series of reasons other AI models gave for a prediction."""
SUMMARY_PROMPT_SUFFIX = """Please return a brief summary of the most common considerations, and whether this made the prediction higher or lower."""

def get_usage(model, result):
  if model.startswith('claude'):
    return {'input': result[-1]['input_tokens'], 'output': result[-1]['output_tokens']}
  elif model.startswith('o1'):
    return {'input': result[-1].prompt_tokens, 'output': result[-1].completion_tokens}
  else:
      raise NotImplementedError("Only Claude and o1 supported for this atm")

COST_DICT = {"claude-3-5-sonnet-20240620": {"input": 3.0/1000000, "output": 15.0/1000000}, "o1-preview": {"input": 15/1000000, "output": 60/1000000}}

def get_cost(model, token_dict):
  if model not in COST_DICT:
    raise NotImplementedError("Only Claude and o1 supported for this atm")
  return token_dict['input'] * COST_DICT[model]['input'] + token_dict['output'] * COST_DICT[model]['output']

async def process_agent(prediction_fn, question_detail, pbar, summary_report, model_fn=call_claude, prompt_template=PROMPT_TEMPLATE):
    result = await prediction_fn(question_detail, summary_report, model_fn, prompt_template)
    pbar.update(1)
    return result

def aggregate_prediction_log_odds(predictions):
    probs = [p / 100 for p in predictions if p is not None]
    probs = [max(min(p, 0.99), 0.01) for p in probs]
    log_odds = [math.log(p / (1 - p)) for p in probs]
    average_log_odds = sum(log_odds) / len(log_odds)
    average_probability = 1 / (1 + math.exp(-average_log_odds))
    return round(average_probability * 100, 0)

def aggregate_prediction_mean(predictions):
    return round(sum(filter(None, predictions)) / len(list(filter(None, predictions))), 0)

def aggregate_by_consensus(predictions):
    # We want the closest to the consensus, which is mean for most values and log odds for extreme values.
    if 10 <= aggregate_prediction_mean(predictions) <= 90:
        return aggregate_prediction_mean(predictions)
    return aggregate_prediction_log_odds(predictions)

async def ensemble_async(prediction_fn, question_ids, num_agents=32, 
                         aggregate_fn = aggregate_prediction_log_odds,
                         news_fn = call_perplexity, model_fn = call_claude, prompt_template = PROMPT_TEMPLATE):
    question_details = [get_question_details(question_id) for question_id in question_ids]
    total_cost = 0
    final_predictions = []
    summaries = []
    model_name = "claude-3-5-sonnet-20241022" if model_fn == call_claude else "o1-preview"
    total_iterations = len(question_details) * num_agents

    with tqdm(total=total_iterations) as pbar:
        for i, question_detail in enumerate(question_details):
            summary_report = await news_fn(question_detail["title"])
            logger.info(f"Question {i+1} of {len(question_details)}: {question_detail['id']} - {question_detail['title']}")
            tasks = []
            for _ in range(num_agents):
                task = asyncio.create_task(process_agent(prediction_fn, question_detail, pbar, summary_report, model_fn, prompt_template))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            predictions, usages, response_texts = [result[0] for result in results], [get_usage(model_name, result) for result in results], [result[2] for result in results]
            final_predictions.append(predictions)
            costs = [get_cost(model_name, usage) for usage in usages]
            total_cost += sum(costs)
            summary, _ = await model_fn(SUMMARY_PROMPT_PREFIX + '\n'.join(response_texts) + SUMMARY_PROMPT_SUFFIX)
            summaries.append(summary)

            logger.info(f"Predictions: {predictions}")
            aggregated_prediction = aggregate_fn(predictions)
            logger.info(f"Aggregated prediction: {aggregated_prediction}")

            if aggregated_prediction is not None and SUBMIT_PREDICTION:
                comment = f"This prediction was made by averaging an ensemble of {num_agents} agents. A summary of their most common considerations follows.\n\n{summaries[i]}"
                logger.info(f"Comment: {comment}")
                post_question_prediction(question_ids[i], aggregated_prediction)
                post_question_comment(question_ids[i], comment)
                logger.info(f"Prediction submitted for question {question_ids[i]}")

    logger.info(f"Total cost was ${round(total_cost, 2)}")

DEBUG_MODE = False
SUBMIT_PREDICTION = not DEBUG_MODE

# TODO: Incorporate TOURNAMENT_ID, API_BASE_URL, and USER_ID as env variables into the code.
def benchmark_all_hyperparameters(ids):
    news_fns = [call_perplexity, call_ask_news]
    model_fns = [call_claude, call_gpt]
    prompts = [PROMPT_TEMPLATE, SUPERFORECASTING_TEMPLATE]

    hyperparams = itertools.product(news_fns, model_fns, prompts)

    for hyperparam in hyperparams:
        logger.info(f"Using hyperparameters: {hyperparam[0], hyperparam[1], 'PROMPT_TEMPLATE' if hyperparam[2] == PROMPT_TEMPLATE else 'SUPERFORECASTING_TEMPLATE'}")
        results = asyncio.run(ensemble_async(get_prediction, ids, num_agents=2, news_fn=hyperparam[0], model_fn=hyperparam[1], prompt_template=hyperparam[2]))
        logger.info(results)
        #logger.info(f"Score: {score_benchmark_results(results)}")

def main():
    data = list_questions(tournament_id=32506, count=2 if DEBUG_MODE else 99, get_answered_questions=DEBUG_MODE)
    ids = [question["id"] for question in data["results"]]
    logger.info(f"Questions found: {ids}")
    if DEBUG_MODE:
        logger.info("WARNING: DEBUG MODE ENABLED. PREDICTIONS WILL NOT BE SUBMITTED.")
    results = asyncio.run(ensemble_async(get_prediction, ids, num_agents=2 if DEBUG_MODE else 32, aggregate_fn=aggregate_by_consensus, news_fn=call_ask_news, model_fn=call_claude, prompt_template=SUPERFORECASTING_TEMPLATE))
    logger.info(results)
    if DEBUG_MODE:
        logger.info("WARNING: DEBUG MODE ENABLED. PREDICTIONS WILL NOT BE SUBMITTED.")
    #benchmark_all_hyperparameters(ids)

if __name__ == "__main__":
    main()