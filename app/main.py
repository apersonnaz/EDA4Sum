import json
import traceback
import aiofiles
from os import stat
from typing import Dict, List, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.summary_evaluator import SummaryEvaluator
from rl.A3C_2_actors.target_set_generator import TargetSetGenerator
from starlette.responses import FileResponse

from app.model_manager import ModelManager
from .format_helper import FormatHelper
from .models import (ByFacetBody, ByFilterBody, ByJoinBody, ByNeighborsBody,
                     ByOverlapBody, DatabaseName, OperatorRequestBody,
                     OperatorRequestResponse, SetDefinition)
from .pipelines.pipeline import Pipeline
from .pipelines.pipeline_precalculated_sets import \
    PipelineWithPrecalculatedSets
from .pipelines.pipeline_sql import PipelineSql
from .pipelines.predicateitem import JoinParameters
from .greedy_summarizer import GreedySummarizer

app = FastAPI(title="CNRS pipelines API",
              description="API providing access to the CNRS pipelines operators",
              version="1.0.0",)

data_folder = "./app/data/"
database_pipeline_cache = {}
database_pipeline_cache["galaxies"] = PipelineWithPrecalculatedSets(
    "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10,
    exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"], id_column="galaxies.objID")
# "dm-authors.gender",  "dm-authors.country",
# columns = ["dm-authors.seniority", "dm-authors.nb_publi",
#            "dm-authors.pub_rate", "dm-authors.first_author_por", "dm-authors.avg_coauthor",
#            "dm-authors.CIKM", "dm-authors.ICDE", "dm-authors.ICWSM", "dm-authors.IEEE", "dm-authors.RecSys",
#            "dm-authors.SIGIR", "dm-authors.SIGMOD", "dm-authors.VLDB", "dm-authors.WSDM", "dm-authors.WWW"]
# database_pipeline_cache["dm-authors"] = PipelineWithPrecalculatedSets(
#     "dm-authors", ["dm-authors"], data_folder=data_folder, discrete_categories_count=5, min_set_size=10, exploration_columns=columns)

# database_pipeline_cache["unics_cordis"] = PipelineSql(
#     "unics_cordis", data_folder=data_folder, discrete_categories_count=10)
# database_pipeline_cache["sdss"] = PipelineSql(
#     "sdss", data_folder=data_folder, discrete_categories_count=10)

model_manager = ModelManager(database_pipeline_cache["galaxies"])


app.mount("/dora-summaries",
          StaticFiles(directory="client", html=True), name="client")


@app.get("/")
async def read_index():
    return FileResponse('./client/redirect.html')


def getPipeline(request: OperatorRequestBody):
    return database_pipeline_cache[request.database]


def get_items_sets(sets, pipeline, get_scores, get_predicted_scores, galaxy_class_scores, seen_sets=None, previous_dataset_ids=None, utility_weights=None, previous_operations=None, decreasing_gamma=False):
    results = {
        "sets": [],
        "previous_operations": previous_operations
    }
    evaluator = SummaryEvaluator(
        pipeline, galaxy_class_scores=galaxy_class_scores)
    evaluator.evaluate_sets(sets)
    results.update(evaluator.get_evaluation_scores())
    results["galaxy_class_scores"] = evaluator.galaxy_class_scores
    for dataset in sets:
        res = {
            "length": len(dataset.data),
            "id": int(dataset.set_id) if dataset.set_id != None else -1,
            "data": [],
            "predicate": []
        }

        for predicate in dataset.predicate.components:
            res["predicate"].append(
                {"dimension": predicate.attribute, "value": str(predicate.value)})

        # set.data = set.data[["galaxies.ra", "galaxies.dec"]]
        if pipeline.database_name == 'sdss':
            if len(dataset.data) > 12:
                data = dataset.data.sample(n=12, random_state=1)
            else:
                data = dataset.data
            for index, galaxy in data[["galaxies.ra", "galaxies.dec"]].iterrows():
                res["data"].append(
                    {"ra": float(galaxy["galaxies.ra"]), "dec": float(galaxy["galaxies.dec"])})
        else:
            data = dataset.data.sort_values(
                "dm-authors.seniority_original", ascending=False)
            if len(dataset.data) > 40:
                data = data.iloc[0:40]
            for index, galaxy in data.iterrows():
                res["data"].append(
                    {"author_name": galaxy["dm-authors.author_name"]})

        results["sets"].append(res)
    if get_scores:
        summary_uniformity_score, sets_uniformity_scores = pipeline.utility_manager.get_uniformity_scores(
            sets, pipeline)
        results["distance"] = pipeline.utility_manager.get_min_distance(
            sets, pipeline)
        results["uniformity"] = summary_uniformity_score

        for index, score in enumerate(sets_uniformity_scores):
            results["sets"][index]["uniformity"] = score
        summary_novelty_score, seen_sets, new_utility_weights = pipeline.utility_manager.get_novelty_scores_and_utility_weights(
            sets, seen_sets, pipeline, utility_weights=utility_weights, decreasing_gamma=decreasing_gamma)
        results["novelty"] = summary_novelty_score
        results["utility"] = pipeline.utility_manager.compute_utility(utility_weights, results["uniformity"],
                                                                      results["distance"], results["novelty"])
        results["utility_weights"] = utility_weights = new_utility_weights
        results["seen_sets"] = seen_sets
    else:
        results["uniformity"] = None
        results["novelty"] = None

        for dataset in results["sets"]:
            dataset["uniformity"] = None
            dataset["novelty"] = None
        seen_sets = seen_sets | set(map(lambda x: int(x.set_id), sets))
        results["seen_sets"] = seen_sets
    if get_predicted_scores:
        results["predictedScores"] = pipeline.utility_manager.get_future_scores(
            sets, pipeline, seen_sets, previous_dataset_ids, utility_weights, previous_operations)
    else:
        results["predictedScores"] = {}

    return results


class OperatorRequest(BaseModel):
    dataset_to_explore: str
    input_set_id: Optional[int] = None
    dimensions: Optional[List[str]] = None
    get_scores: Optional[bool] = False
    get_predicted_scores: Optional[bool] = False
    target_set: Optional[str] = None
    curiosity_weight: Optional[float] = None
    found_items_with_ratio: Optional[Dict[str, float]] = None
    target_items: Optional[List[str]] = None
    previous_set_states: Optional[List[List[float]]] = None
    previous_operation_states: Optional[List[List[float]]] = None
    seen_predicates: Optional[List[str]] = []
    dataset_ids: Optional[List[int]] = None
    seen_sets: Optional[List[int]] = [],
    utility_weights: Optional[List[float]] = [0.333, 0.333, 0.334],
    previous_operations: Optional[List[str]] = [],
    decreasing_gamma: Optional[bool] = False,
    galaxy_class_scores: Optional[Dict[str, float]] = None,
    weights_mode: Optional[str] = None


@app.get("/swapSum")
async def swap_sum(dataset_to_explore: str, min_set_size: int, min_uniformity_target: float, result_set_count: int):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
        dataset_to_explore]
    greedy_summarizer = GreedySummarizer(pipeline)
    result_sets = greedy_summarizer.get_summary(
        min_set_size, min_uniformity_target, result_set_count)
    result = get_items_sets(result_sets, pipeline, True, False, None, seen_sets=set(),
                            previous_dataset_ids=set(), utility_weights=[0.5, 0.5, 0], previous_operations=[])

    return result


@app.put("/get_predicted_scores")
async def get_predicted_scores(operator_request: OperatorRequest):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
        operator_request.dataset_to_explore]
    datasets = pipeline.get_groups_as_datasets(operator_request.dataset_ids)

    result = get_items_sets(datasets, pipeline, True, True, None, seen_sets=set(operator_request.seen_sets),
                            previous_dataset_ids=set(operator_request.dataset_ids), utility_weights=operator_request.utility_weights,
                            previous_operations=operator_request.previous_operations)
    return result


@app.put("/operators/by_facet-g",
         description="Groups the input set items by a list of provided attributes and returns the n biggest resulting sets",
         tags=["operators"])
async def by_facet_g(operator_request: OperatorRequest):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
            operator_request.dataset_to_explore]
        if operator_request.input_set_id == -1:
            dataset = pipeline.get_dataset()
        else:
            dataset = pipeline.get_groups_as_datasets(
                [operator_request.input_set_id])[0]
        number_of_groups = 10 if len(operator_request.dimensions) == 1 else 5
        result_sets = pipeline.by_facet(
            dataset=dataset, attributes=operator_request.dimensions, number_of_groups=number_of_groups)
        result_sets = [d for d in result_sets if d.set_id !=
                       None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]
        ####
        # result_sets = pipeline.get_groups_as_datasets([272574, 326166, 346950, 267949, 346137, 36011, 306809, 292659, 308049, 271882])
        ###
        operation_identifier = f"by_facet-{operator_request.dimensions[0]}-{dataset.set_id}"
        if not operation_identifier in operator_request.previous_operations:
            operator_request.previous_operations.append(operation_identifier)
        prediction_result = {}
        if operator_request.weights_mode != None:
            prediction_result = model_manager.get_prediction(result_sets, operator_request.weights_mode, operator_request.target_items,
                                                             operator_request.found_items_with_ratio, operator_request.previous_set_states, operator_request.previous_operation_states)
        result = get_items_sets(result_sets, pipeline, operator_request.get_scores,
                                operator_request.get_predicted_scores, operator_request.galaxy_class_scores, seen_sets=set(
                                    operator_request.seen_sets),
                                previous_dataset_ids=set(operator_request.dataset_ids), utility_weights=operator_request.utility_weights,
                                previous_operations=operator_request.previous_operations, decreasing_gamma=operator_request.decreasing_gamma)
        result.update(prediction_result)
        return result
    except Exception as error:
        print(error)
        # print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        try:
            print(json.dumps(result))
        except Exception as err:
            print(err)
        return 0


@app.put("/operators/by_superset-g",
         description="Returns the smallest set completely overget-dataset-informationlapping with the input set",
         tags=["operators"])
async def by_superset_g(operator_request: OperatorRequest):
    result = []
    try:
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
            operator_request.dataset_to_explore]
        if operator_request.input_set_id == -1:
            dataset = pipeline.get_dataset()
        else:
            dataset = pipeline.get_groups_as_datasets(
                [operator_request.input_set_id])[0]
        result_sets = pipeline.by_superset(
            dataset=dataset)
        result_sets = [d for d in result_sets if d.set_id !=
                       None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]
        prediction_result = {}

        operation_identifier = f"by_superset--{dataset.set_id}"
        if not operation_identifier in operator_request.previous_operations:
            operator_request.previous_operations.append(operation_identifier)
        if operator_request.weights_mode != None:
            prediction_result = model_manager.get_prediction(result_sets, operator_request.weights_mode, operator_request.target_items,
                                                             operator_request.found_items_with_ratio, operator_request.previous_set_states, operator_request.previous_operation_states)

        result = get_items_sets(result_sets, pipeline, operator_request.get_scores,
                                operator_request.get_predicted_scores, operator_request.galaxy_class_scores, seen_sets=set(
                                    operator_request.seen_sets),
                                previous_dataset_ids=set(operator_request.dataset_ids), utility_weights=operator_request.utility_weights,
                                previous_operations=operator_request.previous_operations, decreasing_gamma=operator_request.decreasing_gamma)

        result.update(prediction_result)
        return result
    except Exception as error:
        print(error)
        # print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        try:
            print(json.dumps(result))
        except Exception as err:
            print(err)
        return 0


@app.put("/operators/by_neighbors-g",
         description="",
         tags=["operators"])
async def by_neighbors_g(operator_request: OperatorRequest):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
            operator_request.dataset_to_explore]
        dataset = pipeline.get_groups_as_datasets(
            [operator_request.input_set_id])[0]
        result_sets = pipeline.by_neighbors(
            dataset=dataset, attributes=operator_request.dimensions)
        result_sets = [d for d in result_sets if d.set_id !=
                       None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]

        prediction_result = {}

        operation_identifier = f"by_neighbors-{operator_request.dimensions[0]}-{dataset.set_id}"
        if not operation_identifier in operator_request.previous_operations:
            operator_request.previous_operations.append(operation_identifier)
        if operator_request.weights_mode != None:
            prediction_result = model_manager.get_prediction(result_sets, operator_request.weights_mode, operator_request.target_items,
                                                             operator_request.found_items_with_ratio, operator_request.previous_set_states, operator_request.previous_operation_states)

        result = get_items_sets(result_sets, pipeline, operator_request.get_scores,
                                operator_request.get_predicted_scores, operator_request.galaxy_class_scores, seen_sets=set(
                                    operator_request.seen_sets),
                                previous_dataset_ids=set(operator_request.dataset_ids), utility_weights=operator_request.utility_weights,
                                previous_operations=operator_request.previous_operations, decreasing_gamma=operator_request.decreasing_gamma)

        result.update(prediction_result)
        return result
    except Exception as error:
        print(error)
        # print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        try:
            print(json.dumps(result))
        except Exception as err:
            print(err)
        return 0


@app.put("/operators/by_distribution-g",
         description="",
         tags=["operators"])
async def by_distribution_g(operator_request: OperatorRequest):
    result = []
    try:
        # print(requestBody.json())
        pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[
            operator_request.dataset_to_explore]
        dataset = pipeline.get_groups_as_datasets(
            [operator_request.input_set_id])[0]
        result_sets = pipeline.by_distribution(
            dataset=dataset)
        result_sets = [d for d in result_sets if d.set_id !=
                       None and d.set_id >= 0]
        if len(result_sets) == 0:
            result_sets = [dataset]
        prediction_result = {}
        operation_identifier = f"by_distribution--{dataset.set_id}"
        if not operation_identifier in operator_request.previous_operations:
            operator_request.previous_operations.append(operation_identifier)
        if operator_request.weights_mode != None:
            prediction_result = model_manager.get_prediction(result_sets, operator_request.weights_mode, operator_request.target_items,
                                                             operator_request.found_items_with_ratio, operator_request.previous_set_states, operator_request.previous_operation_states)
        # result = get_galaxies_sets(result_sets, pipeline, galaxy_request.get_scores,
        #                            galaxy_request.get_predicted_scores, seen_predicates=set(galaxy_request.seen_predicates), previous_dataset_ids=set(galaxy_request.dataset_ids))
        result = get_items_sets(result_sets, pipeline, operator_request.get_scores,
                                operator_request.get_predicted_scores, operator_request.galaxy_class_scores, seen_sets=set(
                                    operator_request.seen_sets),
                                previous_dataset_ids=set(operator_request.dataset_ids), utility_weights=operator_request.utility_weights,
                                previous_operations=operator_request.previous_operations, decreasing_gamma=operator_request.decreasing_gamma)

        result.update(prediction_result)
        return result
    except Exception as error:
        print(error)
        # print(requestBody.json())
        traceback.print_tb(error.__traceback__)
        try:
            print(json.dumps(result))
        except Exception as err:
            print(err)
        return 0


@app.get("/app/get-dataset-information",
         description="",
         tags=["info"])
async def get_dataset_information(dataset: str):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache[dataset]
    return {
        "dimensions": list(pipeline.ordered_dimensions.keys()),
        "ordered_dimensions": pipeline.ordered_dimensions,
        "length": len(pipeline.initial_collection)
    }


@app.get("/app/get-target-items-and-prediction",
         description="",
         tags=["info"])
async def get_target_items_and_prediction(target_set: str = None, curiosity_weight: float = None, dataset_ids: List[int] = []):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    target_items = TargetSetGenerator.get_diverse_target_set(pipeline.database_name,
                                                             number_of_samples=100)
    items_found_with_ratio = {}
    if len(dataset_ids) == 0:
        datasets = [pipeline.get_dataset()]
    else:
        datasets = pipeline.get_groups_as_datasets(dataset_ids)
    if curiosity_weight != None:
        prediction_results = model_manager.get_prediction(
            datasets, target_set, curiosity_weight, target_items, items_found_with_ratio)
        prediction_results["targetItems"] = list(
            map(lambda x: str(x), target_items))
    return prediction_results


@app.put("/app/load-model",
         description="",
         tags=["info"])
async def load_model(galaxy_request: OperatorRequest):
    pipeline: PipelineWithPrecalculatedSets = database_pipeline_cache["galaxies"]
    if len(galaxy_request.dataset_ids) == 0:
        datasets = [pipeline.get_dataset()]
    else:
        datasets = pipeline.get_groups_as_datasets(galaxy_request.dataset_ids)

    prediction_results = model_manager.get_prediction(datasets, galaxy_request.weights_mode, galaxy_request.target_items, galaxy_request.found_items_with_ratio,
                                                      previous_set_states=galaxy_request.previous_set_states, previous_operation_states=galaxy_request.previous_operation_states)

    return prediction_results
