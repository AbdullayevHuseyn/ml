import os
import sys
import requests

# ==== CONFIG - tailored for your setup ====
BASE_URL = "https://webui.huuradar.org"        # Your Open WebUI via Cloudflare
MODEL = "qwen3-coder:30b"                             # Change if you use a different model name
KNOWLEDGE_ID = "c015359c-a841-40b0-8618-cd52824feda7"  # ML_Huseyn collection ID
# ==========================================


def get_api_key():
    api_key = "sk-c5411f5bb66a4b2988014639710a7c63"
    return api_key
def upload_file_to_session(file_path, api_key, session_files):
    """
    Upload a local file and attach it to this chat session.
    Uses Open WebUI /api/v1/files/ endpoint.
    """
    url = f"{BASE_URL}/api/v1/files/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    with open(file_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(url, headers=headers, files=files, timeout=300)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("File upload failed.")
        print("Status:", resp.status_code)
        print("Response text:", resp.text)
        raise

    data = resp.json()

    # Open WebUI returns the file id in "id" for uploaded files
    file_id = data["id"]
    session_files.append({"type": "file", "id": file_id})
    print(f"Uploaded and attached file to this chat. file_id={file_id}")


def ask_openwebui(messages, api_key, session_files):
    """
    Send the conversation + RAG config to Open WebUI /api/chat/completions.
    Always includes your ML_Huseyn collection plus any uploaded files.
    """
    url = f"{BASE_URL}/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Base RAG collection + any per-session files
    files_param = [{"type": "collection", "id": KNOWLEDGE_ID}] + session_files

    payload = {
        "model": MODEL,
        "messages": messages,
        "files": files_param,
        "stream": False,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    # On errors, print server response so we can see what WebUI complains about
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("Server returned an error status code.")
        print("Status:", resp.status_code)
        print("Response text:", resp.text)
        raise

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    api_key = get_api_key()
    print("Connected to ML_Huseyn RAG chat.")
    print("Commands:")
    print("  /upload <path>   - upload a file and attach it to this chat")
    print("  /exit            - quit\n")

    messages = []
    session_files = []  # holds {"type": "file", "id": "..."} for this chat session

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        cmd = user_input.strip()

        # Exit commands
        if cmd.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break

        # Upload command
        if cmd.lower().startswith("/upload "):
            path = user_input[len("/upload "):].strip().strip('"')
            if not path:
                print("Usage: /upload C:\\path\\to\\file.pdf")
                continue
            try:
                upload_file_to_session(path, api_key, session_files)
            except FileNotFoundError:
                print(f"File not found: {path}")
            except Exception as e:
                print(f"Error uploading file: {e}")
            continue

        # Normal chat message
        messages.append({"role": "user", "content": user_input})

        try:
            answer = ask_openwebui(messages, api_key, session_files)
        except Exception as e:
            print(f"Error talking to server: {e}")
            # roll back last message on error
            messages.pop()
            continue

        print("\nAssistant:")
        print(answer)
        print()
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
