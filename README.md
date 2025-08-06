# 🚀 Advanced Gemini Prompting Studio v2.1

An enhanced Streamlit application for advanced prompt engineering with Google Gemini AI, featuring parallel execution, Langfuse tracking, and comprehensive prompt analysis tools.

## ✨ New Features in v2.1

### ⚡ Parallel Execution
- **Simultaneous prompt processing** for faster results
- **Comparative analysis** across different approaches
- **Performance metrics** and execution time tracking
- **Template-based** parallel prompt generation

### 📊 Langfuse Integration
- **Comprehensive tracking** of all AI interactions
- **Real-time observability** and analytics
- **Direct dashboard links** from the UI
- **Detailed trace analysis** for debugging

### 🔗 UI Enhancement
- **Langfuse dashboard links** throughout the application
- **Per-request tracking toggles** for fine-grained control
- **Enhanced analytics** with tracking metrics
- **Seamless integration** with existing workflows

## 🎯 Core Features

### 🔍 Chain of Verification (CoV)
- Improves response accuracy through systematic verification
- Self-generated verification questions
- Joint and factored verification modes
- Template-based verification prompts

### 🧠 Prompting Techniques
- **Zero-Shot**: Direct prompting without examples
- **Few-Shot**: Learning from provided examples
- **Chain-of-Thought**: Step-by-step reasoning
- **Role-Based**: Expert persona prompting
- **ReAct**: Reasoning and Acting combined
- **Instruction-Style**: Direct instruction following

### 🔄 Prompt Engineering Tools
- **Automatic technique detection** and analysis
- **Cross-technique conversion** with context awareness
- **Prompt quality analysis** and improvement suggestions
- **Template library** with customizable variables

### 🧪 Testing & Evaluation
- **Batch testing** with CSV upload
- **Manual testing** with comparison metrics
- **Response quality** analysis
- **Performance benchmarking**

### 🧠 Multimodal Processing
- **Image analysis** with text prompts
- **Chain of Verification** for visual understanding
- **Multiple image** processing support
- **Voice prompt** integration (framework ready)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- Langfuse account (optional but recommended)

### Setup Instructions

1. **Clone or download** the application files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Required API Keys

#### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Enter the key in the sidebar under "Gemini API Key"

#### Langfuse Setup (Recommended)
1. Sign up at [Langfuse Cloud](https://cloud.langfuse.com) or self-host
2. Create a new project
3. Copy your **Secret Key** and **Public Key**
4. Enter them in the sidebar under "Langfuse Configuration"
5. Set your **Host URL** (default: https://cloud.langfuse.com)

### Model Parameters
- **Temperature**: Controls randomness (0.0-2.0)
- **Max Output Tokens**: Response length limit (100-8192)
- **Top P**: Nucleus sampling parameter (0.0-1.0)

## 📖 Usage Guide

### 🎯 Basic Prompting
1. Select a **prompting technique** from the dropdown
2. Fill in the required fields for your chosen technique
3. Enable **Langfuse tracking** (recommended)
4. Click **"Generate Response"**
5. View results and access **Langfuse dashboard** via the provided link

### ⚡ Parallel Execution
1. Navigate to the **"Parallel Execution"** tab
2. Choose your input method:
   - **Manual Entry**: Add prompts individually
   - **Template-based**: Use predefined templates
   - **CSV Upload**: Bulk import prompts
3. Configure **parallel workers** (1-10)
4. Click **"Execute Parallel Prompts"**
5. Analyze **performance metrics** and compare results

### 📊 Langfuse Analytics
- **Dashboard Access**: Click Langfuse links throughout the UI
- **Trace Analysis**: View detailed execution traces
- **Performance Metrics**: Monitor response times and quality
- **Error Tracking**: Debug failed requests

### 🔍 Chain of Verification
1. Enter your query in the **CoV tab**
2. Choose **verification style** (joint/factored)
3. Optionally use **templates** for common scenarios
4. Review the **4-step verification process**:
   - Baseline response generation
   - Verification question creation
   - Question answering
   - Final verified response

### 🔄 Prompt Conversion
1. Enter your **original prompt**
2. Select **source** and **target** techniques
3. Provide **context** (domain, complexity, audience)
4. Generate **converted prompt**
5. Optionally **test both versions** for comparison

## 📊 Template Library

### Available Categories
- **Zero-Shot**: Direct prompts with variables
- **Few-Shot**: Example-based templates
- **Chain-of-Thought**: Step-by-step reasoning
- **Role-Based**: Expert persona prompts
- **ReAct**: Reasoning and acting combined
- **Chain-of-Verification**: Accuracy-focused prompts
- **Parallel**: Multi-perspective analysis templates

### Template Usage
1. Select a **category** and **template**
2. Fill in **template variables**
3. Generate the **final prompt**
4. Execute or use in other features

## 🧪 Testing & Evaluation

### Batch Testing
1. Prepare a **CSV file** with columns: `prompt`, `expected`
2. Upload via the **file uploader**
3. Enable **Langfuse tracking**
4. Run **batch tests**
5. Review **evaluation metrics**

### Manual Testing
1. Enter a **test prompt**
2. Provide **expected output**
3. Run **single test**
4. Compare **actual vs expected** results
5. View **similarity scores**

## 📈 Analytics Dashboard

### Metrics Available
- **Total prompts** processed
- **Techniques used** diversity
- **Average response length**
- **Langfuse tracking** statistics
- **Usage trends** over time
- **Performance analysis** by technique

### Visualizations
- **Technique usage** pie chart
- **Daily usage trends** line graph
- **Response length** distribution
- **Langfuse tracking** breakdown

## 📁 Data Management

### Export Options
- **JSON**: Complete data with metadata
- **CSV**: Tabular format for analysis
- **TXT**: Human-readable format

### Import Features
- **History restoration** from exported files
- **Bulk prompt import** via CSV
- **Template sharing** capabilities

## 🧠 Multimodal Features

### Image Analysis
1. Upload **image files** (JPG, PNG)
2. Provide **text prompts**
3. Optionally apply **Chain of Verification**
4. Process **multiple images** simultaneously
5. View **comprehensive analysis**

### Voice Integration (Framework)
- **Speech recognition** framework ready
- **Voice prompt** input support
- **Transcription** display area

## 🔧 Advanced Configuration

### Parallel Execution Settings
- **Max Workers**: Control concurrent processing
- **Trace Naming**: Custom trace identification
- **Error Handling**: Robust failure management

### Langfuse Integration
- **Selective Tracking**: Per-request control
- **Custom Metadata**: Enhanced trace information
- **Error Logging**: Comprehensive error capture

## 📚 Best Practices

### Prompt Engineering
1. **Start simple** with zero-shot prompts
2. **Add examples** for complex tasks (few-shot)
3. **Use Chain-of-Thought** for reasoning tasks
4. **Apply verification** for critical information
5. **Test variations** with parallel execution

### Langfuse Usage
1. **Enable tracking** for important workflows
2. **Use descriptive trace names** for easy identification
3. **Monitor performance** regularly via dashboard
4. **Analyze failures** to improve prompts

### Performance Optimization
1. **Batch similar prompts** for parallel execution
2. **Adjust worker count** based on rate limits
3. **Monitor response times** and optimize accordingly
4. **Use templates** for consistent results

## 🚨 Troubleshooting

### Common Issues
- **API Key Errors**: Verify key validity and permissions
- **Langfuse Connection**: Check host URL and credentials
- **Rate Limiting**: Reduce parallel workers or add delays
- **Memory Issues**: Process smaller batches or reduce complexity

### Error Handling
- **Graceful failures** with detailed error messages
- **Automatic retries** for transient issues
- **Comprehensive logging** via Langfuse
- **User-friendly** error reporting

## 🤝 Contributing

### Development Setup
1. **Clone repository** and install dependencies
2. **Set up environment** variables for testing
3. **Follow code style** guidelines
4. **Add tests** for new features

### Feature Requests
- **Parallel execution** enhancements
- **Additional prompting** techniques
- **Extended multimodal** support
- **Enhanced analytics** capabilities

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Links

- **Langfuse Documentation**: [https://langfuse.com/docs](https://langfuse.com/docs)
- **Google Gemini API**: [https://ai.google.dev/](https://ai.google.dev/)
- **Streamlit Documentation**: [https://docs.streamlit.io/](https://docs.streamlit.io/)

---

**🚀 Advanced Gemini Prompting Studio v2.1** - Enhanced with Parallel Execution & Langfuse Integration