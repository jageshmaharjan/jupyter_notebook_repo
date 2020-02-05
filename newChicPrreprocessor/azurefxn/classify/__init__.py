import logging

import azure.functions as func
from . import prediction as pred # predict,  get_query

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    headers = {"content-type" : "application/json", "Access-Control-Allow-Origin": "*"}

    query = req.params.get('query')
    if not query:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            query = req_body.get('query')

    if query:
        response = pred.predict(query) # get_query(query) # predict(query)
        return func.HttpResponse(f'query = {query} \n\nlabel = {response}', headers=headers)
    else:
        return func.HttpResponse(
             "Please pass the query string in the request body",
             status_code=400
        )
