import httpx

print("Uploading files...")
url = "http://127.0.0.1:8000/upload"
files = {
    'files': [
        ('acme_security_policy.pdf', open('data/acme_security_policy.pdf', 'rb'), 'application/pdf'),
        ('policy.docx', open('data/policy.docx', 'rb'), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
        ('backup.txt', open('data/backup.txt', 'rb'), 'text/plain')
    ]
}
data = {
    'workspace_id': 'ws_acme_corp',
    'version': '2026-01'
}

try:
    with httpx.Client() as client:
        response = client.post(url, files=[('files', f) for f in files['files']], data=data, timeout=30.0)
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.json())
except Exception as e:
    print("ERROR:", e)

print("Killing server...")
proc.kill()
