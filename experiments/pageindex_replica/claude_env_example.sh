# Azure OpenAI Configuration
# Get these from your Azure Portal: portal.azure.com
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Rate Limiting Configuration
# Adjust based on your Azure OpenAI quota
# Check quota at: portal.azure.com -> Your OpenAI Resource -> Quotas
MAX_REQUESTS_PER_MINUTE=60
MAX_TOKENS_PER_MINUTE=150000

# Processing Configuration
MAX_PAGES_PER_CHUNK=10
MAX_TOKENS_PER_CHUNK=100000
MAX_CONCURRENT_REQUESTS=5

# How to get your Azure OpenAI credentials:
# 1. Go to https://portal.azure.com
# 2. Navigate to your Azure OpenAI resource
# 3. Go to "Keys and Endpoint" in the left sidebar
# 4. Copy the endpoint URL and one of the keys
# 5. Note your deployment name from "Deployments" section