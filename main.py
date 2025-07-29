import os
import json
import sys
import boto3
print("imported successfully...")

prompt = """

    you are a smart assistant so please let me know what is machine learning in smartest way?


"""
bedrock=boto3.client(service_name="bedrock-runtime")
payload={


}

body= json.dumps(payload)
model_id="meta.llama2-70b-chat-v1"

response = bedrock.invoke_model(
    body=body,
    model_id=model_id,
    accept="application/json",
    content_type="application/json")
response_body=json.loads(response.get("body").read())
response_test= response_body["generation"]
print(response_test)
