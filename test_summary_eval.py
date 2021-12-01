from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from app.summary_evaluator import SummaryEvaluator

data_folder = "./app/data/"
pipeline = PipelineWithPrecalculatedSets(
    "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])

summary_evaluator = SummaryEvaluator(pipeline)
summary_evaluator.evaluate_sets(
    pipeline.get_groups_as_datasets([452, 254869, 246077]))
print('oh')
