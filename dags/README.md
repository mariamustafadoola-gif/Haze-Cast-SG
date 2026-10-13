Airflow DAGs go here (Lecture 6). Planned:
- hazecast_data_dag: daily 06:00, ingest > bronze > silver > checks > gold (data team)
- hazecast_training_dag: monthly, train > evaluate > register (AI team)
- hazecast_forecast_dag: daily 06:00 after the data DAG, score > forecast table > monitoring (AI team)
