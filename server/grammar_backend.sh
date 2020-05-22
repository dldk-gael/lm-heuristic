# LAUNCH CELERY WORKER WHICH HANDLE TASK THAT NEED TO USE LANGUAGE MODEL
# BECAUSE LANGUAGE MODEL TAKE A LOT OF SPACE IN MEMORY, WE DO NOT ALLOWED CONCURRENY 
# ELSE IT WOULD HAVE LOADED SEVERAL TIME GPT2 / BERT MODEL IN MEMORY 
celery worker -A grammar_backend.celery --loglevel=info -Q language_model --concurrency 1 &

# LAUNCH CELERY WORKER WHICH HANDLE THE OTHER TASKS
celery worker -A grammar_backend.celery --loglevel=info  -Q default &

# LAUNCH FLASK SERVER
python run.py &

# LAUNCH REDIS SERVER (HANDLE COMMUNICATION BETWEEN FLASK AND WORKERS)
redis-server & 

