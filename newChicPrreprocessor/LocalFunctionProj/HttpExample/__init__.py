import logging

import azure.functions as func
# import HttpExample/prediction
from . import prediction as pred #import get_query

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        res = pred.predict(name)
        return func.HttpResponse(f"Your query = {name} \n\nLabel = {res}!")
    else:
        return func.HttpResponse(
             "Please pass a name on the query string or in the request body",
             status_code=400
        )
