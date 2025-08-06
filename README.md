# 🚀 Advanced Prompting Studio with Langfuse

A professional-grade Streamlit application for advanced prompting techniques with parallel execution and comprehensive tracking powered by Langfuse.

## ✨ Features

### 🎯 Advanced Prompting Techniques
- **Zero-Shot Prompting** - Direct questions without examples
- **Few-Shot Prompting** - Learning from examples
- **Chain-of-Thought** - Step-by-step reasoning
- **Role-Based Prompting** - Expert persona adoption
- **Chain-of-Verification (CoV)** - Enhanced accuracy through verification

### ⚡ Parallel Execution
- **Concurrent Processing** - Execute multiple prompts simultaneously
- **Batch Analysis** - Compare different approaches efficiently
- **A/B Testing** - Test prompt variations at scale
- **Template-Based Generation** - Automated prompt variations
- **File Upload Support** - Bulk prompt processing from CSV

### 📊 Langfuse Integration
- **Complete Observability** - Track every prompt execution
- **Performance Analytics** - Detailed metrics and insights
- **Error Monitoring** - Comprehensive error tracking
- **Execution Traces** - Step-by-step execution visibility
- **Dashboard Links** - Direct access to Langfuse analytics

### 🧠 Additional Features
- **Multimodal Prompting** - Text + image processing
- **Template Library** - Pre-built prompt templates
- **Analytics Dashboard** - Usage patterns and trends
- **Export/Import** - Data portability (JSON, CSV, TXT)
- **Prompt Converter** - Transform between techniques

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key
- Langfuse account (optional but recommended)

### Quick Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd advanced-prompting-studio
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🔧 Configuration

### API Keys

#### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Enter the key in the sidebar "Gemini API Key" field

#### Langfuse (Optional)
1. Sign up at [langfuse.com](https://langfuse.com)
2. Create a new project
3. Get your Public Key and Secret Key from project settings
4. Enter the keys in the sidebar "Langfuse Tracking" section
5. Enable tracking with the checkbox

### Model Configuration
- **Temperature**: Controls randomness (0.0-2.0)
- **Max Tokens**: Maximum response length (100-8192)
- **Top P**: Nucleus sampling parameter (0.0-1.0)

### Parallel Execution Settings
- **Max Workers**: Number of concurrent executions (1-10)
- **Timeout**: Maximum execution time per prompt (10-120s)

## 📖 Usage Guide

### Single Prompt Execution

1. **Select a Technique** from the dropdown
2. **Fill in the required fields** based on the technique
3. **Click "Generate Response"** to execute
4. **View results** and Langfuse trace (if enabled)

### Parallel Execution

1. **Navigate to the "Parallel Execution" tab**
2. **Choose setup method:**
   - **Manual Entry**: Configure individual prompts
   - **Template-Based**: Generate variations from a template
   - **File Upload**: Bulk upload from CSV file
3. **Configure execution settings** (workers, timeout)
4. **Click "Execute in Parallel"** to run all prompts
5. **View comprehensive results** with individual traces

### Chain of Verification

1. **Navigate to "Chain of Verification" tab**
2. **Enter your query** or use a template
3. **Choose verification style** (joint/factored)
4. **Configure advanced options** (optional)
5. **Run the verification process** and view step-by-step results

## 📊 Langfuse Integration

### Benefits
- **Complete Observability**: Every prompt execution is tracked
- **Performance Insights**: Detailed analytics on usage patterns
- **Error Monitoring**: Comprehensive error tracking and debugging
- **Optimization**: Identify the best-performing prompts and techniques

### Dashboard Features
- **Traces**: Detailed execution logs with timing and parameters
- **Analytics**: Usage statistics and performance metrics
- **Models**: Track different model performance
- **Datasets**: Organize and version your prompts

### Accessing Langfuse
- **Direct Links**: Click the "View in Langfuse Dashboard" buttons
- **Main Dashboard**: Use the "Langfuse Insights" tab
- **Quick Access**: Sidebar links when tracking is enabled

## 🔬 Prompting Techniques

### Zero-Shot Prompting
Direct questioning without examples. Best for:
- Simple tasks
- General knowledge questions
- Creative writing

### Few-Shot Prompting
Learning from examples. Best for:
- Classification tasks
- Pattern recognition
- Specific formatting requirements

### Chain-of-Thought
Step-by-step reasoning. Best for:
- Mathematical problems
- Complex reasoning
- Multi-step analysis

### Role-Based Prompting
Adopting expert personas. Best for:
- Domain-specific tasks
- Professional advice
- Specialized knowledge

### Chain-of-Verification
Enhanced accuracy through verification. Best for:
- Factual claims
- Research tasks
- Critical information

## ⚡ Parallel Processing

### Use Cases
- **A/B Testing**: Compare prompt variations
- **Batch Processing**: Handle large datasets
- **Technique Comparison**: Evaluate different approaches
- **Performance Testing**: Assess execution efficiency

### Setup Methods
1. **Manual Entry**: Individual prompt configuration
2. **Template-Based**: Automated variations with variables
3. **File Upload**: CSV bulk processing

### Benefits
- **Speed**: 3-5x faster than sequential execution
- **Efficiency**: Maximize API utilization
- **Scalability**: Handle large prompt volumes
- **Insights**: Compare multiple approaches simultaneously

## 📈 Analytics & Insights

### Built-in Analytics
- **Usage Statistics**: Technique frequency and patterns
- **Performance Metrics**: Response times and success rates
- **Response Analysis**: Length distribution and characteristics
- **Model Comparison**: Different model performance

### Langfuse Analytics
- **Advanced Metrics**: Detailed performance insights
- **Cost Tracking**: Monitor API usage and costs
- **Quality Scores**: Prompt effectiveness evaluation
- **Trend Analysis**: Usage patterns over time

## 🎨 Templates

### Available Categories
- **Zero-Shot**: Translation, summarization, Q&A, code generation
- **Few-Shot**: Classification, NER, code review
- **Chain-of-Thought**: Math problems, logic puzzles, reasoning
- **Role-Based**: Expert consultation, teaching, analysis
- **Chain-of-Verification**: Factual claims, research, explanations

### Custom Templates
Create and save your own templates for repeated use.

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify your Gemini API key is valid
   - Check API quotas and limits
   - Ensure proper network connectivity

2. **Langfuse Connection Issues**
   - Verify public and secret keys
   - Check Langfuse host URL
   - Ensure proper network access

3. **Parallel Execution Errors**
   - Reduce number of workers
   - Increase timeout settings
   - Check individual prompt validity

4. **Memory Issues**
   - Reduce batch sizes
   - Clear history periodically
   - Monitor system resources

### Performance Tips

- **Use parallel execution** for multiple prompts
- **Enable Langfuse tracking** for insights
- **Optimize prompt templates** for efficiency
- **Monitor execution times** and adjust accordingly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web app framework
- **Google Gemini** - For powerful AI capabilities
- **Langfuse** - For comprehensive observability
- **Plotly** - For beautiful visualizations

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Review Langfuse documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Built with ❤️ using Streamlit, Google Gemini API, and Langfuse**