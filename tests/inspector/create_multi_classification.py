from .create_binary_classification import df, prediction_pipeline, selected_columns, target

target_modified = target.copy()
target_modified[target_modified.sample(frac=0.1).index] = 2

prediction_pipeline.fit(df[selected_columns], target_modified)
