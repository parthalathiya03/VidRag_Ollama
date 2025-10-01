#!/bin/bash

echo "🚀 Setting up Fully Local Voice RAG Assistant..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}✅ Detected macOS${NC}"
else
    echo -e "${BLUE}ℹ️  Detected: $OSTYPE${NC}"
fi

# Create necessary directories
echo ""
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p docs
mkdir -p models
mkdir -p vector_store

# Create sample documents
echo ""
echo -e "${BLUE}📄 Creating sample documents...${NC}"
cat > docs/doc1.txt << 'EOF'
Python is a high-level, interpreted programming language created by Guido van Rossum. It was first released in 1991. Python emphasizes code readability with its notable use of significant indentation. It supports multiple programming paradigms including structured, object-oriented, and functional programming.
EOF

cat > docs/doc2.txt << 'EOF'
Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn from and make predictions or decisions based on data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.
EOF

cat > docs/doc3.txt << 'EOF'
Streamlit is an open-source Python library that makes it easy to create beautiful web apps for machine learning and data science. It was created by Streamlit Inc. and allows developers to turn data scripts into shareable web apps in minutes without requiring front-end development experience.
EOF

echo -e "${GREEN}✅ Created 3 sample documents in docs/${NC}"

# Check for Python
echo ""
echo -e "${BLUE}🐍 Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check for pip
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✅ pip3 is installed${NC}"
else
    echo -e "${RED}❌ pip3 not found${NC}"
    exit 1
fi

# Install Python dependencies
echo ""
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip3 install -r requirements.txt

# Download embedding model
echo ""
echo -e "${BLUE}🤖 Downloading embedding model (first time only)...${NC}"
python3 << 'EOF'
from sentence_transformers import SentenceTransformer
print("Downloading all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model downloaded!")
EOF

# Check for Ollama
echo ""
echo -e "${BLUE}🦙 Checking Ollama installation...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✅ Ollama is installed${NC}"
    
    # Check if llama3 model is installed
    if ollama list | grep -q "llama3"; then
        echo -e "${GREEN}✅ llama3 model is already installed${NC}"
    else
        echo -e "${BLUE}📥 Downloading llama3 model (this may take a few minutes)...${NC}"
        ollama pull llama3
        echo -e "${GREEN}✅ llama3 model downloaded${NC}"
    fi
else
    echo -e "${RED}❌ Ollama not found${NC}"
    echo ""
    echo "Please install Ollama:"
    echo "1. Visit: https://ollama.ai"
    echo "2. Download and install for macOS"
    echo "3. Run: ollama pull llama3"
    echo ""
    echo "After installing Ollama, run this script again."
    exit 1
fi

# Create .env file if it doesn't exist
echo ""
if [ ! -f .env ]; then
    echo -e "${BLUE}📝 Creating .env file...${NC}"
    cat > .env << 'EOF'
# API Keys
DEEPGRAM_API_KEY=your_deepgram_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
EOF
    echo -e "${GREEN}✅ Created .env file${NC}"
    echo -e "${RED}⚠️  Please edit .env and add your API keys!${NC}"
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your Deepgram and ElevenLabs API keys"
echo "2. Make sure Ollama is running: ollama serve"
echo "3. Run the app: streamlit run app.py"
echo ""
echo "📚 Sample documents are in the docs/ folder"
echo "🎉 Everything runs locally except speech services!"
echo ""