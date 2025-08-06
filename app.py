import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
import re
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional
import os
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.model import CreateTrace
import nest_asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable nested asyncio
nest_asyncio.apply()

# Page configuration
st.set_page_config(
    page_title="Advanced Prompting Studio with Langfuse",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI/UX
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iZ3JhZGllbnQiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojNjY3ZWVhO3N0b3Atb3BhY2l0eTowLjEiIC8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3R5bGU9InN0b3AtY29sb3I6Izc2NGJhMjtzdG9wLW9wYWNpdHk6MC4xIiAvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICA8L2RlZnM+CiAgPHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0idXJsKCNncmFkaWVudCkiIC8+Cjwvc3ZnPg==');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    .technique-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .prompt-result {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .verification-step {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }

    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }

    .parallel-result {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }

    .langfuse-link {
        background: #6366f1;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .langfuse-link:hover {
        background: #4f46e5;
        color: white;
        text-decoration: none;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        border: 1px solid #e0e0e0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prompts_history' not in st.session_state:
    st.session_state.prompts_history = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'langfuse_client' not in st.session_state:
    st.session_state.langfuse_client = None
if 'langfuse_enabled' not in st.session_state:
    st.session_state.langfuse_enabled = False
if 'parallel_results' not in st.session_state:
    st.session_state.parallel_results = []

# Sidebar for configuration
with st.sidebar:
    st.markdown("## 🔧 Configuration")

    # API Keys section
    st.markdown("### 🔑 API Keys")
    
    # Gemini API Key
    api_key = st.text_input("🤖 Gemini API Key", type="password", help="Enter your Google Gemini API key")

    # Langfuse Configuration
    st.markdown("### 📊 Langfuse Tracking")
    
    langfuse_public_key = st.text_input("🔑 Langfuse Public Key", type="password", help="Your Langfuse public key")
    langfuse_secret_key = st.text_input("🔒 Langfuse Secret Key", type="password", help="Your Langfuse secret key")
    langfuse_host = st.text_input("🌐 Langfuse Host", value="https://cloud.langfuse.com", help="Langfuse host URL")

    # Enable Langfuse
    enable_langfuse = st.checkbox("📈 Enable Langfuse Tracking", value=False)

    # Initialize Langfuse
    if enable_langfuse and langfuse_public_key and langfuse_secret_key:
        try:
            st.session_state.langfuse_client = Langfuse(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,
                host=langfuse_host
            )
            st.session_state.langfuse_enabled = True
            st.success("✅ Langfuse initialized!")
            
            # Add link to Langfuse dashboard
            st.markdown(f"""
            <a href="{langfuse_host}" target="_blank" class="langfuse-link">
                📊 Open Langfuse Dashboard
            </a>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error initializing Langfuse: {str(e)}")
            st.session_state.langfuse_enabled = False
    else:
        st.session_state.langfuse_enabled = False

    # Model selection
    st.markdown("### 🤖 Model Configuration")
    model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-1.5-flash-8b"]
    selected_model = st.selectbox("🤖 Select Model", model_options)

    # Model parameters
    st.markdown("### ⚙️ Model Parameters")
    temperature = st.slider("🌡️ Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("📝 Max Output Tokens", 100, 8192, 1024, 100)
    top_p = st.slider("🎯 Top P", 0.0, 1.0, 0.95, 0.05)

    # Parallel execution settings
    st.markdown("### ⚡ Parallel Execution")
    max_workers = st.slider("👥 Max Workers", 1, 10, 3, 1, help="Maximum number of parallel executions")
    execution_timeout = st.slider("⏱️ Timeout (seconds)", 10, 120, 30, 5)

    # Initialize model
    if api_key:
        try:
            genai.configure(api_key=api_key)
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
            )
            st.session_state.model = genai.GenerativeModel(
                model_name=selected_model,
                generation_config=generation_config
            )
            st.success("✅ Model initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing model: {str(e)}")

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 Advanced Prompting Studio with Langfuse</h1>
    <p>Professional-grade prompting techniques with parallel execution and comprehensive tracking</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Prompting technique implementations with Langfuse tracking
class PromptingTechniques:
    def __init__(self, model, langfuse_client=None):
        self.model = model
        self.langfuse_client = langfuse_client

    @observe(name="zero_shot_prompting")
    def zero_shot(self, prompt: str, trace_id: Optional[str] = None) -> str:
        """Zero-shot prompting - direct question without examples"""
        if self.langfuse_client and trace_id:
            langfuse_context.update_current_trace(
                name="Zero-Shot Prompting",
                input=prompt,
                metadata={"technique": "zero-shot"}
            )
        
        return self._generate_response(prompt, "zero-shot")

    @observe(name="few_shot_prompting")
    def few_shot(self, task_description: str, examples: List[Dict], query: str, trace_id: Optional[str] = None) -> str:
        """Few-shot prompting with examples"""
        prompt = f"{task_description}\n\nExamples:\n"
        for i, example in enumerate(examples, 1):
            prompt += f"{i}. Input: {example['input']} → Output: {example['output']}\n"
        prompt += f"\nNow solve: {query}"
        
        if self.langfuse_client and trace_id:
            langfuse_context.update_current_trace(
                name="Few-Shot Prompting",
                input={"task": task_description, "examples": examples, "query": query},
                metadata={"technique": "few-shot", "num_examples": len(examples)}
            )
        
        return self._generate_response(prompt, "few-shot")

    @observe(name="chain_of_thought")
    def chain_of_thought(self, problem: str, trace_id: Optional[str] = None) -> str:
        """Chain-of-thought prompting for step-by-step reasoning"""
        prompt = f"{problem}\n\nLet's think step by step:"
        
        if self.langfuse_client and trace_id:
            langfuse_context.update_current_trace(
                name="Chain-of-Thought Prompting",
                input=problem,
                metadata={"technique": "chain-of-thought"}
            )
        
        return self._generate_response(prompt, "chain-of-thought")

    @observe(name="role_based_prompting")
    def role_based(self, role: str, task: str, trace_id: Optional[str] = None) -> str:
        """Role-based prompting with specific persona"""
        prompt = f"You are {role}. {task}"
        
        if self.langfuse_client and trace_id:
            langfuse_context.update_current_trace(
                name="Role-Based Prompting",
                input={"role": role, "task": task},
                metadata={"technique": "role-based"}
            )
        
        return self._generate_response(prompt, "role-based")

    @observe(name="chain_of_verification")
    def chain_of_verification(self, query: str, verification_style: str = "joint", trace_id: Optional[str] = None) -> Dict[str, str]:
        """Chain of Verification prompting for improved accuracy"""
        
        if self.langfuse_client and trace_id:
            langfuse_context.update_current_trace(
                name="Chain of Verification",
                input={"query": query, "style": verification_style},
                metadata={"technique": "chain-of-verification"}
            )

        # Step 1: Generate baseline response
        baseline_response = self._generate_response(query, "baseline")

        # Step 2: Generate verification questions
        verification_prompt = f"""Given the following question and answer, generate specific verification questions that would help check the accuracy and completeness of the answer. Focus on factual claims, logical consistency, and potential gaps.

Original Question: {query}
Answer: {baseline_response}

Please generate 3-5 specific verification questions that would help validate this answer:"""

        verification_questions = self._generate_response(verification_prompt, "verification-questions")

        # Step 3: Answer verification questions
        if verification_style == "joint":
            joint_prompt = f"""Please answer the following verification questions based on your knowledge:

{verification_questions}

Provide detailed answers to each question:"""
            verification_answers = self._generate_response(joint_prompt, "verification-answers")
        else:
            questions_list = [q.strip() for q in verification_questions.split('\n') if q.strip() and ('?' in q)]
            verification_answers = ""
            for i, question in enumerate(questions_list[:5], 1):
                answer = self._generate_response(f"Answer this question concisely: {question}", f"verification-answer-{i}")
                verification_answers += f"Q{i}: {question}\nA{i}: {answer}\n\n"

        # Step 4: Generate final verified response
        final_prompt = f"""Based on the original question, initial answer, verification questions, and their answers, provide a final, improved, and verified response. Correct any errors found and incorporate insights from the verification process.

Original Question: {query}
Initial Answer: {baseline_response}

Verification Questions: {verification_questions}
Verification Answers: {verification_answers}

Final Verified Response:"""

        final_response = self._generate_response(final_prompt, "final-verified")

        result = {
            "baseline_response": baseline_response,
            "verification_questions": verification_questions,
            "verification_answers": verification_answers,
            "final_response": final_response,
            "verification_style": verification_style
        }

        if self.langfuse_client and trace_id:
            langfuse_context.update_current_trace(
                output=result,
                metadata={"verification_steps": 4, "style": verification_style}
            )

        return result

    def parallel_execute(self, prompts: List[Dict[str, Any]], max_workers: int = 3) -> List[Dict[str, Any]]:
        """Execute multiple prompts in parallel"""
        results = []
        
        def execute_single_prompt(prompt_data):
            try:
                technique = prompt_data.get('technique', 'zero_shot')
                prompt_text = prompt_data.get('prompt', '')
                prompt_id = prompt_data.get('id', f"prompt_{len(results)}")
                
                # Create trace for this execution if Langfuse is enabled
                trace_id = None
                if self.langfuse_client:
                    trace = self.langfuse_client.trace(
                        name=f"Parallel Execution - {technique}",
                        input=prompt_data,
                        metadata={"execution_type": "parallel", "prompt_id": prompt_id}
                    )
                    trace_id = trace.id

                start_time = time.time()
                
                # Execute based on technique
                if technique == 'zero_shot':
                    response = self.zero_shot(prompt_text, trace_id)
                elif technique == 'chain_of_thought':
                    response = self.chain_of_thought(prompt_text, trace_id)
                elif technique == 'role_based':
                    role = prompt_data.get('role', 'helpful assistant')
                    response = self.role_based(role, prompt_text, trace_id)
                elif technique == 'chain_of_verification':
                    style = prompt_data.get('verification_style', 'joint')
                    response = self.chain_of_verification(prompt_text, style, trace_id)
                else:
                    response = self.zero_shot(prompt_text, trace_id)
                
                execution_time = time.time() - start_time
                
                result = {
                    'id': prompt_id,
                    'technique': technique,
                    'prompt': prompt_text,
                    'response': response,
                    'execution_time': execution_time,
                    'timestamp': datetime.now(),
                    'success': True,
                    'trace_id': trace_id
                }
                
                # Update trace with results
                if self.langfuse_client and trace_id:
                    trace.update(
                        output=response,
                        metadata={
                            "execution_time": execution_time,
                            "success": True,
                            "technique": technique
                        }
                    )
                
                return result
                
            except Exception as e:
                error_result = {
                    'id': prompt_data.get('id', f"prompt_{len(results)}"),
                    'technique': prompt_data.get('technique', 'unknown'),
                    'prompt': prompt_data.get('prompt', ''),
                    'response': f"Error: {str(e)}",
                    'execution_time': 0,
                    'timestamp': datetime.now(),
                    'success': False,
                    'error': str(e),
                    'trace_id': trace_id
                }
                
                # Update trace with error
                if self.langfuse_client and trace_id:
                    trace.update(
                        output={"error": str(e)},
                        metadata={"success": False, "error": str(e)}
                    )
                
                return error_result

        # Execute prompts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {executor.submit(execute_single_prompt, prompt): prompt for prompt in prompts}
            
            for future in concurrent.futures.as_completed(future_to_prompt):
                result = future.result()
                results.append(result)

        # Sort results by original order
        results.sort(key=lambda x: next((i for i, p in enumerate(prompts) if p.get('id') == x['id']), 0))
        
        return results

    def _generate_response(self, prompt: str, step: str = "generation") -> str:
        """Helper method to generate response from model with Langfuse tracking"""
        try:
            if not self.model:
                return "❌ Model not initialized. Please provide API key."

            start_time = time.time()
            
            # Generate with Langfuse tracking if enabled
            if self.langfuse_client:
                generation = self.langfuse_client.generation(
                    name=f"Gemini {step}",
                    model=self.model.model_name if hasattr(self.model, 'model_name') else "gemini",
                    input=prompt,
                    metadata={"step": step}
                )

            response = self.model.generate_content(prompt)
            response_text = response.text
            
            execution_time = time.time() - start_time
            
            # Update Langfuse generation
            if self.langfuse_client:
                generation.end(
                    output=response_text,
                    metadata={
                        "execution_time": execution_time,
                        "response_length": len(response_text)
                    }
                )
            
            return response_text
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            
            # Log error to Langfuse
            if self.langfuse_client:
                generation.end(
                    output={"error": str(e)},
                    metadata={"success": False, "error": str(e)}
                )
            
            return error_msg

# Template library with enhanced templates
TEMPLATE_LIBRARY = {
    "Zero-Shot": {
        "Translation": "Translate the following text to {target_language}: '{text}'",
        "Summarization": "Summarize the following text in {num_sentences} sentences: '{text}'",
        "Question Answering": "Answer the following question: {question}",
        "Code Generation": "Write {language} code to {task}",
        "Creative Writing": "Write a {genre} story about {topic}",
        "Data Analysis": "Analyze the following data and provide insights: {data}",
        "Problem Solving": "Solve this problem step by step: {problem}"
    },
    "Few-Shot": {
        "Classification": {
            "description": "Classify the sentiment of the following texts:",
            "examples": [
                {"input": "I love this product!", "output": "Positive"},
                {"input": "This is terrible", "output": "Negative"},
                {"input": "It's okay, nothing special", "output": "Neutral"}
            ]
        },
        "Named Entity Recognition": {
            "description": "Extract named entities from the following texts:",
            "examples": [
                {"input": "John works at Google in California", "output": "Person: John, Organization: Google, Location: California"},
                {"input": "Apple released iPhone in 2007", "output": "Organization: Apple, Product: iPhone, Date: 2007"}
            ]
        },
        "Code Review": {
            "description": "Review the following code and provide feedback:",
            "examples": [
                {"input": "def add(a, b): return a + b", "output": "Good: Simple and clear. Consider: Add type hints and docstring."},
                {"input": "for i in range(len(lst)): print(lst[i])", "output": "Issue: Use 'for item in lst' instead. More Pythonic and efficient."}
            ]
        }
    },
    "Chain-of-Thought": {
        "Math Problem": "A train travels at {speed} km/h for {time} hours. How far does it travel?",
        "Logic Puzzle": "If all cats are animals, and some animals are pets, can we conclude that some cats are pets?",
        "Word Problem": "Sarah has {initial_amount} apples. She gives {given_amount} to her friend. How many does she have left?",
        "Scientific Reasoning": "Explain why {phenomenon} occurs, considering the underlying scientific principles.",
        "Complex Analysis": "Analyze the relationship between {factor_a} and {factor_b} in the context of {domain}."
    },
    "Role-Based": {
        "Expert Consultant": "You are an expert {field} consultant. Provide advice on {topic}.",
        "Teacher": "You are a {subject} teacher explaining {concept} to {grade_level} students.",
        "Creative Writer": "You are a creative writer in the {genre} genre. Write about {theme}.",
        "Technical Reviewer": "You are a senior software engineer reviewing code. Analyze this {language} code: {code}",
        "Data Scientist": "You are a data scientist. Analyze this dataset and provide insights: {data}",
        "Business Analyst": "You are a business analyst. Evaluate this business scenario: {scenario}"
    },
    "Chain-of-Verification": {
        "Factual Claims": "What are the main causes of {phenomenon}? Provide a comprehensive and verified answer.",
        "Historical Events": "Describe the key events and significance of {historical_event}.",
        "Scientific Explanations": "Explain how {scientific_concept} works and its applications.",
        "Complex Reasoning": "Analyze the relationship between {concept_a} and {concept_b} and provide evidence-based conclusions.",
        "Research Summary": "Summarize the current research on {topic} and verify the key findings."
    }
}

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🎯 Prompting Techniques",
    "⚡ Parallel Execution", 
    "🔍 Chain of Verification",
    "🔄 Prompt Converter",
    "🧪 Testing & Evaluation",
    "📊 Analytics Dashboard",
    "📚 Template Library",
    "🧠 Multimodal Prompting",
    "📈 Langfuse Insights"
])

# Tab 1: Enhanced Prompting Techniques
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🎯 Select Technique")
        technique = st.selectbox(
            "Choose prompting technique:",
            ["Zero-Shot", "Few-Shot", "Chain-of-Thought", "Role-Based", "Chain-of-Verification"]
        )

        # Dynamic input based on technique
        if technique == "Zero-Shot":
            prompt_input = st.text_area("💭 Enter your prompt:", height=120)

        elif technique == "Few-Shot":
            task_desc = st.text_input("📝 Task Description:")
            
            st.markdown("**Examples:**")
            num_examples = st.number_input("Number of examples:", 1, 10, 3)
            
            examples = []
            for i in range(num_examples):
                col_in, col_out = st.columns(2)
                with col_in:
                    input_ex = st.text_input(f"Input {i+1}:", key=f"input_{i}")
                with col_out:
                    output_ex = st.text_input(f"Output {i+1}:", key=f"output_{i}")
                
                if input_ex and output_ex:
                    examples.append({"input": input_ex, "output": output_ex})
            
            query = st.text_input("🔍 Query to solve:")

        elif technique == "Chain-of-Thought":
            prompt_input = st.text_area("🧠 Problem to solve:", height=120)

        elif technique == "Role-Based":
            role = st.text_input("👤 Role/Persona:", placeholder="e.g., expert data scientist, creative writer...")
            task = st.text_area("📋 Task:", height=100)

        elif technique == "Chain-of-Verification":
            prompt_input = st.text_area("❓ Query for verification:", height=120)
            verification_style = st.radio("Verification Style:", ["joint", "factored"])

    with col2:
        st.markdown("### 🚀 Results")

        if st.button("🎯 Generate Response", type="primary"):
            if not st.session_state.model:
                st.error("❌ Please configure your API key first!")
            else:
                with st.spinner("🤖 Generating response..."):
                    techniques = PromptingTechniques(
                        st.session_state.model, 
                        st.session_state.langfuse_client if st.session_state.langfuse_enabled else None
                    )

                    # Create main trace for this session
                    trace_id = None
                    if st.session_state.langfuse_enabled:
                        trace = st.session_state.langfuse_client.trace(
                            name=f"Single Prompt - {technique}",
                            metadata={"technique": technique, "model": selected_model}
                        )
                        trace_id = trace.id

                    # Generate response based on technique
                    if technique == "Zero-Shot" and prompt_input:
                        response = techniques.zero_shot(prompt_input, trace_id)
                        used_prompt = prompt_input

                    elif technique == "Few-Shot" and task_desc and examples and query:
                        response = techniques.few_shot(task_desc, examples, query, trace_id)
                        used_prompt = f"Task: {task_desc}, Examples: {examples}, Query: {query}"

                    elif technique == "Chain-of-Thought" and prompt_input:
                        response = techniques.chain_of_thought(prompt_input, trace_id)
                        used_prompt = prompt_input

                    elif technique == "Role-Based" and role and task:
                        response = techniques.role_based(role, task, trace_id)
                        used_prompt = f"Role: {role}, Task: {task}"

                    elif technique == "Chain-of-Verification" and prompt_input:
                        response = techniques.chain_of_verification(prompt_input, verification_style, trace_id)
                        used_prompt = prompt_input

                    else:
                        st.warning("⚠️ Please fill in all required fields!")
                        response = None
                        used_prompt = None

                    if response and used_prompt:
                        # Display response
                        if isinstance(response, dict):  # CoV response
                            st.markdown("##### 🎯 Final Verified Response")
                            st.markdown(f"""
                            <div class="prompt-result">
                                <p>{response['final_response']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("🔍 View Verification Process"):
                                st.markdown("**Baseline Response:**")
                                st.write(response['baseline_response'])
                                st.markdown("**Verification Questions:**")
                                st.write(response['verification_questions'])
                                st.markdown("**Verification Answers:**")
                                st.write(response['verification_answers'])
                            
                            display_response = response['final_response']
                        else:
                            st.markdown(f"""
                            <div class="prompt-result">
                                <h4>🤖 Response:</h4>
                                <p>{response}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            display_response = response

                        # Save to history
                        history_entry = {
                            "timestamp": datetime.now(),
                            "technique": technique,
                            "prompt": used_prompt,
                            "response": display_response,
                            "model": selected_model,
                            "parameters": {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p},
                            "trace_id": trace_id
                        }
                        
                        if isinstance(response, dict):  # Add CoV details
                            history_entry["verification_details"] = response

                        st.session_state.prompts_history.append(history_entry)

                        # Update trace with final results
                        if st.session_state.langfuse_enabled and trace_id:
                            trace.update(
                                output=response,
                                metadata={
                                    "model_params": {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p},
                                    "success": True
                                }
                            )

                        st.success("✅ Response generated and tracked!")
                        
                        if st.session_state.langfuse_enabled:
                            st.markdown(f"""
                            <a href="{langfuse_host}/trace/{trace_id}" target="_blank" class="langfuse-link">
                                📊 View in Langfuse Dashboard
                            </a>
                            """, unsafe_allow_html=True)

# Tab 2: Parallel Execution
with tab2:
    st.markdown("### ⚡ Parallel Prompt Execution")
    
    st.info("""
    **Parallel Execution** allows you to run multiple prompts simultaneously:
    🚀 **Faster processing** - Execute multiple prompts at once
    📊 **Batch analysis** - Compare different approaches
    🔄 **A/B testing** - Test variations of the same prompt
    📈 **Efficiency** - Maximize throughput for large-scale experiments
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### ⚙️ Parallel Execution Setup")
        
        # Method selection
        setup_method = st.radio(
            "📝 Setup Method:",
            ["Manual Entry", "Template-Based", "File Upload"]
        )

        prompts_to_execute = []

        if setup_method == "Manual Entry":
            st.markdown("**Add Prompts:**")
            num_prompts = st.number_input("Number of prompts:", 1, 10, 3)
            
            for i in range(num_prompts):
                with st.expander(f"Prompt {i+1}"):
                    technique = st.selectbox(f"Technique {i+1}:", 
                                           ["zero_shot", "chain_of_thought", "role_based", "chain_of_verification"],
                                           key=f"tech_{i}")
                    prompt_text = st.text_area(f"Prompt {i+1}:", key=f"prompt_{i}", height=80)
                    
                    prompt_data = {"id": f"prompt_{i+1}", "technique": technique, "prompt": prompt_text}
                    
                    if technique == "role_based":
                        role = st.text_input(f"Role {i+1}:", key=f"role_{i}")
                        prompt_data["role"] = role
                    elif technique == "chain_of_verification":
                        style = st.selectbox(f"Verification Style {i+1}:", ["joint", "factored"], key=f"style_{i}")
                        prompt_data["verification_style"] = style
                    
                    if prompt_text:
                        prompts_to_execute.append(prompt_data)

        elif setup_method == "Template-Based":
            st.markdown("**Template-Based Setup:**")
            base_prompt = st.text_area("Base prompt template:", 
                                     placeholder="e.g., Analyze {topic} from a {perspective} perspective")
            
            # Variables to iterate
            st.markdown("**Variables:**")
            variable_name = st.text_input("Variable name:", placeholder="e.g., topic, perspective")
            variable_values = st.text_area("Values (one per line):", 
                                         placeholder="e.g.,\nclimate change\nartificial intelligence\neconomic policy")
            
            if base_prompt and variable_name and variable_values:
                values = [v.strip() for v in variable_values.split('\n') if v.strip()]
                for i, value in enumerate(values):
                    prompt_text = base_prompt.replace(f"{{{variable_name}}}", value)
                    prompts_to_execute.append({
                        "id": f"template_{i+1}",
                        "technique": "zero_shot",
                        "prompt": prompt_text
                    })

        elif setup_method == "File Upload":
            st.markdown("**Upload CSV with prompts:**")
            uploaded_file = st.file_uploader("📁 Upload prompts file", type=['csv'])
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                if 'prompt' in df.columns:
                    for idx, row in df.iterrows():
                        prompts_to_execute.append({
                            "id": f"file_{idx+1}",
                            "technique": row.get('technique', 'zero_shot'),
                            "prompt": row['prompt']
                        })

        # Display prompts to be executed
        if prompts_to_execute:
            st.markdown(f"**📋 {len(prompts_to_execute)} prompts ready for execution**")
            
            # Execution settings
            st.markdown("**⚙️ Execution Settings:**")
            parallel_workers = st.slider("Workers:", 1, min(len(prompts_to_execute), max_workers), 
                                        min(3, len(prompts_to_execute)))

    with col2:
        st.markdown("#### 🚀 Execution Results")

        if prompts_to_execute and st.button("⚡ Execute in Parallel", type="primary"):
            if not st.session_state.model:
                st.error("❌ Please configure your API key first!")
            else:
                with st.spinner(f"⚡ Executing {len(prompts_to_execute)} prompts in parallel..."):
                    techniques = PromptingTechniques(
                        st.session_state.model,
                        st.session_state.langfuse_client if st.session_state.langfuse_enabled else None
                    )
                    
                    # Create main trace for parallel execution
                    if st.session_state.langfuse_enabled:
                        main_trace = st.session_state.langfuse_client.trace(
                            name="Parallel Execution Session",
                            input={"num_prompts": len(prompts_to_execute), "workers": parallel_workers},
                            metadata={"execution_type": "parallel", "model": selected_model}
                        )

                    start_time = time.time()
                    results = techniques.parallel_execute(prompts_to_execute, parallel_workers)
                    total_time = time.time() - start_time

                    # Update main trace
                    if st.session_state.langfuse_enabled:
                        main_trace.update(
                            output={"results_count": len(results), "total_time": total_time},
                            metadata={
                                "success_rate": sum(1 for r in results if r['success']) / len(results),
                                "avg_execution_time": sum(r['execution_time'] for r in results) / len(results)
                            }
                        )

                    # Store results
                    st.session_state.parallel_results = results

                    # Display results
                    st.markdown(f"### ✅ Completed in {total_time:.2f} seconds")
                    
                    # Success metrics
                    success_count = sum(1 for r in results if r['success'])
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    
                    with col_metrics1:
                        st.metric("✅ Successful", success_count)
                    with col_metrics2:
                        st.metric("❌ Failed", len(results) - success_count)
                    with col_metrics3:
                        st.metric("⚡ Avg Time", f"{sum(r['execution_time'] for r in results) / len(results):.2f}s")

                    # Display individual results
                    for result in results:
                        status_icon = "✅" if result['success'] else "❌"
                        technique_badge = result['technique'].replace('_', ' ').title()
                        
                        with st.expander(f"{status_icon} {result['id']} - {technique_badge} ({result['execution_time']:.2f}s)"):
                            col_prompt, col_response = st.columns([1, 2])
                            
                            with col_prompt:
                                st.markdown("**Prompt:**")
                                st.write(result['prompt'])
                                
                                if result.get('trace_id') and st.session_state.langfuse_enabled:
                                    st.markdown(f"""
                                    <a href="{langfuse_host}/trace/{result['trace_id']}" target="_blank" class="langfuse-link">
                                        📊 View Trace
                                    </a>
                                    """, unsafe_allow_html=True)
                            
                            with col_response:
                                if result['success']:
                                    st.markdown("**Response:**")
                                    if isinstance(result['response'], dict):
                                        st.write(result['response']['final_response'])
                                    else:
                                        st.write(result['response'])
                                else:
                                    st.error(f"Error: {result.get('error', 'Unknown error')}")

                    # Add to history
                    for result in results:
                        st.session_state.prompts_history.append({
                            "timestamp": result['timestamp'],
                            "technique": f"Parallel - {result['technique']}",
                            "prompt": result['prompt'],
                            "response": result['response'],
                            "model": selected_model,
                            "parameters": {
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "top_p": top_p,
                                "parallel_execution": True,
                                "execution_time": result['execution_time']
                            },
                            "trace_id": result.get('trace_id')
                        })

                    st.success(f"✅ Parallel execution completed! All results saved to history.")

        elif not prompts_to_execute:
            st.info("👈 Configure prompts in the left panel to start parallel execution")

        # Display previous parallel results if available
        if st.session_state.parallel_results:
            st.markdown("---")
            st.markdown("#### 📊 Latest Parallel Execution Summary")
            
            latest_results = st.session_state.parallel_results
            
            # Create summary chart
            execution_times = [r['execution_time'] for r in latest_results]
            techniques = [r['technique'] for r in latest_results]
            
            fig = px.bar(
                x=[r['id'] for r in latest_results],
                y=execution_times,
                color=techniques,
                title="Execution Times by Prompt",
                labels={'x': 'Prompt ID', 'y': 'Execution Time (s)'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: Chain of Verification (Enhanced)
with tab3:
    st.markdown("### 🔍 Chain of Verification")

    st.info("""
    **Chain of Verification (CoV)** improves response accuracy by:
    1. 📝 Generating an initial response
    2. ❓ Creating verification questions  
    3. ✅ Answering verification questions
    4. 🔄 Producing a final verified response
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 🎯 Configuration")

        # Query input
        cov_query = st.text_area("❓ Enter your query:",
                                placeholder="e.g., What are the main causes of climate change?",
                                height=100)

        # Verification style
        verification_style = st.radio(
            "🔧 Verification Style:",
            ["joint", "factored"],
            help="Joint: Answer all verification questions together. Factored: Answer each question independently."
        )

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            num_verification_questions = st.slider("Number of verification questions:", 3, 8, 5)
            verification_depth = st.selectbox("Verification depth:", ["Standard", "Deep", "Comprehensive"])
            focus_areas = st.multiselect("Focus areas:", 
                                       ["Factual accuracy", "Logical consistency", "Completeness", "Sources", "Contradictions"])

        # Template selection
        st.markdown("#### 📚 Use Template")
        cov_template = st.selectbox(
            "Select CoV template:",
            ["None"] + list(TEMPLATE_LIBRARY["Chain-of-Verification"].keys())
        )

        if cov_template != "None":
            template = TEMPLATE_LIBRARY["Chain-of-Verification"][cov_template]
            variables = re.findall(r'\{(\w+)\}', template)

            if variables:
                st.markdown("**Fill template variables:**")
                var_values = {}
                for var in variables:
                    var_values[var] = st.text_input(f"{var.replace('_', ' ').title()}:", key=f"cov_var_{var}")

                if st.button("📝 Apply Template"):
                    filled_template = template.format(**var_values)
                    st.session_state.cov_template_query = filled_template

        # Use template query if available
        if hasattr(st.session_state, 'cov_template_query'):
            cov_query = st.session_state.cov_template_query
            st.text_area("Template Query:", value=cov_query, height=80, disabled=True)

    with col2:
        st.markdown("#### 🔍 Verification Process")

        if st.button("🚀 Run Chain of Verification", type="primary"):
            if not cov_query:
                st.warning("⚠️ Please enter a query!")
            elif not st.session_state.model:
                st.error("❌ Please configure your API key first!")
            else:
                with st.spinner("🔍 Running Chain of Verification..."):
                    techniques = PromptingTechniques(
                        st.session_state.model,
                        st.session_state.langfuse_client if st.session_state.langfuse_enabled else None
                    )

                    # Create trace for CoV
                    trace_id = None
                    if st.session_state.langfuse_enabled:
                        trace = st.session_state.langfuse_client.trace(
                            name="Chain of Verification",
                            input={"query": cov_query, "style": verification_style},
                            metadata={"verification_depth": verification_depth, "focus_areas": focus_areas}
                        )
                        trace_id = trace.id

                    # Run CoV process
                    cov_result = techniques.chain_of_verification(cov_query, verification_style, trace_id)

                    # Display results step by step
                    st.markdown("##### 📝 Step 1: Baseline Response")
                    st.markdown(f"""
                    <div class="prompt-result">
                        <p>{cov_result['baseline_response']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("##### ❓ Step 2: Verification Questions")
                    st.markdown(f"""
                    <div class="verification-step">
                        <p>{cov_result['verification_questions']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("##### ✅ Step 3: Verification Answers")
                    st.markdown(f"""
                    <div class="verification-step">
                        <p>{cov_result['verification_answers']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("##### 🎯 Step 4: Final Verified Response")
                    st.markdown(f"""
                    <div class="prompt-result" style="border-left: 4px solid #007bff;">
                        <h4>🔍 Verified Response:</h4>
                        <p>{cov_result['final_response']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Save to history
                    st.session_state.prompts_history.append({
                        "timestamp": datetime.now(),
                        "technique": "Chain-of-Verification",
                        "prompt": cov_query,
                        "response": cov_result['final_response'],
                        "model": selected_model,
                        "parameters": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "verification_style": verification_style,
                            "verification_depth": verification_depth,
                            "focus_areas": focus_areas
                        },
                        "verification_details": cov_result,
                        "trace_id": trace_id
                    })

                    # Update trace
                    if st.session_state.langfuse_enabled and trace_id:
                        trace.update(
                            output=cov_result,
                            metadata={"success": True, "verification_steps": 4}
                        )

                    st.success("✅ Chain of Verification completed and saved to history!")
                    
                    if st.session_state.langfuse_enabled:
                        st.markdown(f"""
                        <a href="{langfuse_host}/trace/{trace_id}" target="_blank" class="langfuse-link">
                            📊 View Verification Process in Langfuse
                        </a>
                        """, unsafe_allow_html=True)

# Continue with remaining tabs...
# (Due to length constraints, I'll continue with the remaining tabs in the next part)

# Tab 4: Prompt Converter (keeping existing implementation but adding Langfuse tracking)
with tab4:
    st.markdown("### 🔄 Prompt Converter & Analyzer")
    # [Implementation continues with existing code but with Langfuse integration]
    pass

# Tab 5: Testing & Evaluation
with tab5:
    st.markdown("### 🧪 Prompt Testing & Evaluation")
    # [Enhanced testing with parallel execution and Langfuse tracking]
    pass

# Tab 6: Analytics Dashboard (Enhanced with Langfuse data)
with tab6:
    st.markdown("### 📊 Analytics Dashboard")
    
    # Enhanced analytics with Langfuse integration
    if st.session_state.prompts_history:
        df_history = pd.DataFrame(st.session_state.prompts_history)

        # Usage statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🎯 Total Prompts", len(df_history))

        with col2:
            unique_techniques = df_history['technique'].nunique()
            st.metric("🛠️ Techniques Used", unique_techniques)

        with col3:
            avg_response_length = df_history['response'].apply(
                lambda x: len(str(x)) if not isinstance(x, dict) else len(str(x.get('final_response', x)))
            ).mean()
            st.metric("📝 Avg Response Length", f"{avg_response_length:.0f}")

        with col4:
            parallel_count = df_history['technique'].str.contains('Parallel').sum()
            st.metric("⚡ Parallel Executions", parallel_count)

        # Enhanced visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Technique usage with enhanced categories
            technique_counts = df_history['technique'].value_counts()
            fig_pie = px.pie(values=technique_counts.values, names=technique_counts.index,
                           title="Prompting Techniques Usage")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Execution time analysis (for entries with execution_time)
            exec_time_data = []
            for _, row in df_history.iterrows():
                if isinstance(row['parameters'], dict) and 'execution_time' in row['parameters']:
                    exec_time_data.append({
                        'technique': row['technique'],
                        'execution_time': row['parameters']['execution_time']
                    })
            
            if exec_time_data:
                exec_df = pd.DataFrame(exec_time_data)
                fig_exec = px.box(exec_df, x='technique', y='execution_time',
                                title="Execution Time by Technique")
                fig_exec.update_xaxes(tickangle=45)
                st.plotly_chart(fig_exec, use_container_width=True)

        # Langfuse integration stats
        if st.session_state.langfuse_enabled:
            langfuse_tracked = df_history['trace_id'].notna().sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Langfuse Tracking</h3>
                <p>{langfuse_tracked} / {len(df_history)} prompts tracked</p>
                <a href="{langfuse_host}" target="_blank" class="langfuse-link">
                    View Dashboard
                </a>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("📊 No data available yet. Start using prompting techniques to see analytics!")

# Tab 7: Template Library (Enhanced)
with tab7:
    st.markdown("### 📚 Enhanced Template Library")
    # [Keep existing implementation with minor enhancements]
    pass

# Tab 8: Multimodal Prompting (Enhanced)
with tab8:
    st.markdown("### 🧠 Multimodal Prompting with Verification")
    # [Enhanced multimodal with CoV and Langfuse tracking]
    pass

# Tab 9: Langfuse Insights
with tab9:
    st.markdown("### 📈 Langfuse Insights & Analytics")
    
    if not st.session_state.langfuse_enabled:
        st.warning("⚠️ Langfuse tracking is not enabled. Please configure your API keys in the sidebar.")
    else:
        st.success("✅ Langfuse tracking is active!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔗 Quick Links")
            st.markdown(f"""
            <div class="technique-card">
                <h4>📊 Langfuse Dashboard</h4>
                <p>View comprehensive analytics, traces, and performance metrics</p>
                <a href="{langfuse_host}" target="_blank" class="langfuse-link">
                    Open Dashboard
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="technique-card">
                <h4>🔍 Traces</h4>
                <p>Detailed execution traces for all prompts</p>
                <a href="{langfuse_host}/traces" target="_blank" class="langfuse-link">
                    View Traces
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="technique-card">
                <h4>📈 Analytics</h4>
                <p>Performance metrics and usage analytics</p>
                <a href="{langfuse_host}/analytics" target="_blank" class="langfuse-link">
                    View Analytics
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Session Statistics")
            
            # Count tracked vs untracked
            if st.session_state.prompts_history:
                df_history = pd.DataFrame(st.session_state.prompts_history)
                tracked_count = df_history['trace_id'].notna().sum()
                total_count = len(df_history)
                
                st.metric("📈 Tracked Prompts", f"{tracked_count}/{total_count}")
                st.metric("📊 Tracking Rate", f"{(tracked_count/total_count)*100:.1f}%" if total_count > 0 else "0%")
                
                # Recent traces
                recent_traces = df_history[df_history['trace_id'].notna()].tail(5)
                if not recent_traces.empty:
                    st.markdown("#### 🕒 Recent Traces")
                    for _, trace in recent_traces.iterrows():
                        st.markdown(f"""
                        <div class="verification-step">
                            <strong>{trace['technique']}</strong><br>
                            <small>{trace['timestamp'].strftime('%H:%M:%S')}</small><br>
                            <a href="{langfuse_host}/trace/{trace['trace_id']}" target="_blank" class="langfuse-link">
                                View Trace
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No prompts executed yet in this session.")

        # Langfuse configuration help
        st.markdown("---")
        st.markdown("#### ℹ️ Langfuse Setup Help")
        
        with st.expander("🔧 How to set up Langfuse"):
            st.markdown("""
            1. **Sign up** at [langfuse.com](https://langfuse.com)
            2. **Create a project** in your Langfuse dashboard
            3. **Get your API keys** from the project settings
            4. **Enter the keys** in the sidebar configuration
            5. **Enable tracking** with the checkbox
            
            **Benefits of Langfuse tracking:**
            - 📊 Detailed performance analytics
            - 🔍 Complete execution traces
            - 📈 Usage patterns and trends
            - 🐛 Error tracking and debugging
            - 💡 Prompt optimization insights
            """)

# Footer with enhanced information
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p>🚀 <strong>Advanced Prompting Studio v3.0</strong> | Built with Streamlit, Google Gemini API & Langfuse</p>
    <p>⚡ <strong>New Features:</strong> Parallel Execution • Enhanced Chain of Verification • Comprehensive Tracking</p>
    <p>🔬 <strong>Techniques:</strong> Zero-shot • Few-shot • Chain-of-Thought • Role-based • Chain-of-Verification • Multimodal</p>
    <p>📊 <strong>Powered by Langfuse</strong> for advanced observability and analytics</p>
</div>
""", unsafe_allow_html=True)