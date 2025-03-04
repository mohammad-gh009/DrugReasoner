import os
import traceback
from typing import Dict, List, Union

import requests  # type: ignore
from bs4 import BeautifulSoup
from cache import FileCache
from dotenv import load_dotenv
from download_pmc_s3 import download_pmc_s3
from indra_nxml_extraction import extract_text, id_lookup
from openai import OpenAI

load_dotenv()

subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
endpoint = "https://api.bing.microsoft.com/v7.0/search"


# Create cache instances
search_cache = FileCache("search_results")
dti_score_cache = FileCache("search_dti_scores")


def bing_search(query: str) -> Dict:
    # Check cache
    cached_result = search_cache.get(query)
    if cached_result is not None:
        return cached_result

    mkt = "en-US"
    params = {"q": query, "mkt": mkt}
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        result = response.json()
        # Save result to cache
        search_cache.set(query, result)
        return result
    except requests.RequestException as ex:
        raise ex


def extract_bing_search_results(api_response: Dict) -> List[Dict[str, str]]:
    results = []
    urls = []
    if "webPages" in api_response and "value" in api_response["webPages"]:
        for item in api_response["webPages"]["value"]:
            title = item.get("name", "")
            snippet = item.get("snippet", "")
            url = item.get("url", "")
            urls.append(url)
            results.append({"title": title, "snippet": snippet, "url": url})
    print(f"Extracted URLs: {urls}")
    if urls:
        get_pmid(urls)
    return results


def get_pmid(urls):
    print(urls)
    for url in urls:
        if "nih" in url:
            id = url.rstrip("/").split("/")[-1]
            print("id", id)
            add_new_paper(id)


def read_and_extract_xml_data(fileName):
    with open(fileName, "r") as f:
        data = f.read()
    xml_data = BeautifulSoup(data, "xml")
    text = extract_text(xml_data)
    return text


def add_new_paper(id):
    print(f"Adding new paper with ID: {id}")
    try:
        if id.startswith("PMC"):
            pmcid = id
        else:
            pmcid = id_lookup(id).get("pmcid")
        try:
            download_pmc_s3(pmcid)
            fileName = str("pmc/" + pmcid + ".xml")
            try:
                # read and extract xml data
                text = read_and_extract_xml_data(fileName)
                # write a txt file for extract xml content
                directory_path = "pubmed_paper_fulltext"
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                with open(f"pubmed_paper_fulltext/{pmcid}.txt", "w") as f:
                    f.write(str(text))

            except Exception as e:
                print(f"Error read and extract pmcid {pmcid}: {e}")
        except Exception as e:
            print(f"can't be downloaded {pmcid}: {e}")

    except Exception as e:
        print(f"This {id} has no pmcid: {e}")


def calculate_dti_score(
    search_results: List[Dict[str, str]], drug_name: str, target_name: str
) -> float:
    total_score = 0
    max_score = len(search_results)

    positive_keywords = ["interacts", "binds", "activates", "modulates"]
    strong_keywords = ["strong", "significant", "potent", "effective"]

    for result in search_results:
        score = _calculate_individual_score(
            result, drug_name, target_name, positive_keywords, strong_keywords
        )
        total_score += score

    if max_score == 0:
        return 0.0

    normalized_score = total_score / max_score
    return min(1.0, normalized_score)


def chat_with_gpt(question):
    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY"
        ),  # This is the default and can be omitted
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
    )
    return response.choices[0].message.content


def generate_reasoning(drug_name: str, target_name: str, search_result: Dict) -> str:

    question = (
        f"Based on the following search results for the query about interaction between drug '{drug_name}' and target '{target_name}', "
        f"please provide a reasoning explaining the relationship between them. "
        f"Here are some key snippets from the search results: {search_result}"
    )
    reasoning = chat_with_gpt(question)
    return reasoning


def _calculate_individual_score(
    result: Dict[str, str],
    drug_name: str,
    target_name: str,
    positive_keywords: List[str],
    strong_keywords: List[str],
) -> int:
    score = 0
    text = f"{result['title']} {result['snippet']}".lower()

    if drug_name.lower() in text and target_name.lower() in text:
        score += 1
    if any(keyword in text for keyword in positive_keywords):
        score += 1
    if any(keyword in text for keyword in strong_keywords):
        score += 1

    return score


def search_drug_target(drug_name: str, target_name: str) -> List[Dict[str, str]]:
    query = f"{drug_name} {target_name} interaction"
    return extract_bing_search_results(bing_search(query))


def get_dti_score(name: str, target_name: str) -> tuple[float, str]:
    # Cache key for DTI score
    cache_key = f"{name}_{target_name}"

    # Check cache
    cached_score = dti_score_cache.get(cache_key)
    if cached_score is not None:
        return cached_score["score"], cached_score["reasoning"]

    try:
        search_results = search_drug_target(name, target_name)
        print("search_results is :", search_results)
        # Convert list of dictionaries to a single dictionary for compatibility
        search_results_dict = {"search_results": search_results}
        reasoning = generate_reasoning(name, target_name, search_results_dict)
        dti_score = calculate_dti_score(search_results, name, target_name)
        # Save score to cache
        dti_score_cache.set(cache_key, {"score": dti_score, "reasoning": reasoning})
        return dti_score, reasoning
    except Exception:
        traceback.print_exc()
        return 0.0, "an error occurred during get_dti_score"


def get_dti_scores(
    drugs: List[str], targets: List[str]
) -> List[List[Union[str, float]]]:
    scores: List[List[Union[str, float]]] = []
    try:
        for drug, target in zip(drugs, targets):
            score, reasoning = get_dti_score(drug, target)
            scores.append([drug, target, score, reasoning])
        print("search scores: ", scores)
        return scores
    except Exception as e:
        print(f"An error occurred during the get_dti_scores: {str(e)}")
        traceback.print_exc()
        return scores
