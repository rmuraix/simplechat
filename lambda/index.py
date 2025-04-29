import json
import os
import urllib.request
import urllib.error

INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "https://f5aa-34-16-245-45.ngrok-free.app/generate")

def parse_event(event):
    body = json.loads(event['body'])
    message = body['message']
    history = body.get('conversationHistory', [])
    return message, history

def build_prompt(history, message):
    messages = history.copy()
    messages.append({"role": "user", "content": message})
    
    prompt_lines = []
    for msg in messages:
        prefix = "User:" if msg["role"] == "user" else "Assistant:"
        prompt_lines.append(f"{prefix} {msg['content']}")
    prompt_lines.append("Assistant:")  # モデルの応答を促す
    return "\n".join(prompt_lines), messages

def call_inference_api(prompt):
    payload = {
        "prompt": prompt,
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9
    }

    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        INFERENCE_API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request) as response:
        response_data = response.read().decode('utf-8')
        return json.loads(response_data)

def success_response(body):
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": json.dumps(body)
    }

def error_response(error_message):
    return {
        "statusCode": 500,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": json.dumps({
            "success": False,
            "error": error_message
        })
    }

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))

        # メッセージと履歴の抽出
        message, history = parse_event(event)
        print("User message:", message)

        # プロンプトと更新済み履歴を作成
        prompt, messages = build_prompt(history, message)
        print("Generated prompt:", prompt)

        # 推論APIを呼び出し
        api_response = call_inference_api(prompt)
        generated_text = api_response.get("generated_text", "")
        print("Generated response:", generated_text)

        # 応答を履歴に追加
        messages.append({"role": "assistant", "content": generated_text})

        return success_response({
            "success": True,
            "response": generated_text,
            "conversationHistory": messages
        })

    except (urllib.error.HTTPError, urllib.error.URLError) as net_err:
        print("HTTP error:", str(net_err))
        return error_response(str(net_err))

    except Exception as e:
        print("Unexpected error:", str(e))
        return error_response(str(e))
