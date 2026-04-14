import urllib.request, json

req = urllib.request.urlopen('http://localhost:8000/api/health')
health = json.loads(req.read())
print('Health:', health)

data = json.dumps({
    'age': 28, 'account_balance': 1200.0,
    'transaction_amount': 4500.0,
    'transaction_type': 'Online',
    'merchant_category': 'Electronics',
    'device_type': 'Mobile',
    'transaction_location': 'Nigeria',
    'is_foreign_transaction': 1,
    'failed_attempts_last_24h': 4,
    'avg_transaction_amount': 250.0,
    'transaction_frequency_24h': 12,
    'amount_deviation': 18.0
}).encode()

req2 = urllib.request.Request(
    'http://localhost:8000/api/fraud/analyze',
    data=data, headers={'Content-Type': 'application/json'}, method='POST'
)
result = json.loads(urllib.request.urlopen(req2).read())
print()
print('=== FRAUD ANALYSIS RESULT ===')
print(f'Is Fraud:     {result["is_fraud"]}')
print(f'Risk Score:   {result["risk_percentage"]}%')
print(f'Risk Label:   {result["risk_label"]}')
print(f'Confidence:   {result["confidence"]}%')
print(f'ProcessTime:  {result["processing_time_ms"]}ms')
print(f'Recommendation: {result["recommendation"]}')
