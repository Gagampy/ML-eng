from datetime import timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator

from rtn.airflow_tasks import (
    join_datatables_task,
    split_data_task,
    get_feature_quantiles_task,
    save_splitted_dataset_task,
    remove_outliers_from_train_task,
    remove_outliers_from_valid_task,
    remove_outliers_from_test_task,
    save_train_feature_quantiles
)
from rtn.constants import (
    DATAFOLDER_LOAD_PATH,
    DATAFOLDER_SAVE_PATH,
    RANDOM_SEED,
    TRAIN_RATIO,
    VALID_RATIO,
    UPPER_QUANTILE,
    LOWER_QUANTILE,
)


default_args = {
    "owner": "gagampy",
    "depends_on_past": False,
    "start_date": days_ago(2),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


dag = DAG(
    "rtn_parallel_preprocessing",
    default_args=default_args,
    description="Preprocessing pipeline for RTN",
    schedule_interval=timedelta(days=1),
)

join_datatables_operator = PythonOperator(
    task_id="joining_data",
    provide_context=True,
    op_args=[DATAFOLDER_LOAD_PATH],
    python_callable=join_datatables_task,
    do_xcom_push=True,
    dag=dag,
)

split_data_operator = PythonOperator(
    task_id="splitting_data",
    provide_context=True,
    python_callable=split_data_task,
    do_xcom_push=True,
    op_kwargs={
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "seed": RANDOM_SEED,
    },
    dag=dag,
)

calculating_feature_quantiles_operator = PythonOperator(
    task_id="calculating_feature_quantiles",
    provide_context=True,
    python_callable=get_feature_quantiles_task,
    do_xcom_push=True,
    op_kwargs={"lower_quantile": LOWER_QUANTILE, "upper_quantile": UPPER_QUANTILE},
    dag=dag,
)

remove_outliers_train_operator = PythonOperator(
    task_id="removing_outliers_train",
    provide_context=True,
    python_callable=remove_outliers_from_train_task,
    do_xcom_push=True,
    dag=dag,
)

remove_outliers_valid_operator = PythonOperator(
    task_id="removing_outliers_valid",
    provide_context=True,
    python_callable=remove_outliers_from_valid_task,
    do_xcom_push=True,
    dag=dag,
)

remove_outliers_test_operator = PythonOperator(
    task_id="removing_outliers_test",
    provide_context=True,
    python_callable=remove_outliers_from_test_task,
    do_xcom_push=True,
    dag=dag,
)

save_feature_quantiles_operator = PythonOperator(
    task_id='saving_feature_quantiles',
    provide_context=True,
    python_callable=save_train_feature_quantiles,
    op_kwargs={"savefolder_path": DATAFOLDER_SAVE_PATH},
    do_xcom_push=True,
    dag=dag,
)

save_dataset_operator = PythonOperator(
    task_id="saving_dataset",
    provide_context=True,
    python_callable=save_splitted_dataset_task,
    op_kwargs={
        "savefolder_path": DATAFOLDER_SAVE_PATH,
        "source_task_id": [
            "removing_outliers_train",
            "removing_outliers_valid",
            "removing_outliers_test",
        ],
    },
    dag=dag,
)


join_datatables_operator >> split_data_operator >> calculating_feature_quantiles_operator >> [
    remove_outliers_train_operator,
    remove_outliers_valid_operator,
    remove_outliers_test_operator,
] >> save_dataset_operator

calculating_feature_quantiles_operator.set_downstream(save_feature_quantiles_operator)
