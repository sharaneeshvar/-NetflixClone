# !pip install streamlit google-generativeai pyngrok pandas plotly langfuse
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
from typing import List, Dict, Any
import asyncio
import concurrent.futures
from langfuse import Langfuse
from langfuse.decorators import observe
import threading

# Page configuration
st.set_page_config(
    page_title="Prompting Studio",
    page_icon="🤖",
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
        background-image: url('https://files.worldwildlife.org/wwfcmsprod/images/Tiger_resting_Bandhavgarh_National_Park_India/hero_full/77ic6i4qdj_Medium_WW226365.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    .technique-card {
        background: #020c17;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }

    .prompt-result {
        background: #030c14;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }

    .verification-step {
        background: #0f0d04;
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

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    .parallel-result {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00d2ff;
        margin: 0.5rem 0;
    }

    .langfuse-link {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem 0;
        font-weight: bold;
    }

    .langfuse-link:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,210,255,0.3);
        text-decoration: none;
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
if 'langfuse' not in st.session_state:
    st.session_state.langfuse = None
if 'langfuse_url' not in st.session_state:
    st.session_state.langfuse_url = ""

# Sidebar for configuration
with st.sidebar:
    st.markdown("## 🔧 Configuration")

    # API Key input
    api_key = st.text_input("🔑 Gemini API Key", type="password", help="Enter your Google Gemini API key")

    # Langfuse configuration
    st.markdown("### 📊 Langfuse Configuration")
    langfuse_secret_key = st.text_input("🔐 Langfuse Secret Key", type="password", help="Enter your Langfuse secret key")
    langfuse_public_key = st.text_input("🔑 Langfuse Public Key", help="Enter your Langfuse public key")
    langfuse_host = st.text_input("🌐 Langfuse Host", value="https://cloud.langfuse.com", help="Langfuse host URL")

    # Initialize Langfuse
    if langfuse_secret_key and langfuse_public_key:
        try:
            st.session_state.langfuse = Langfuse(
                secret_key=langfuse_secret_key,
                public_key=langfuse_public_key,
                host=langfuse_host
            )
            st.session_state.langfuse_url = f"{langfuse_host.rstrip('/')}/project/{langfuse_public_key}"
            st.success("✅ Langfuse initialized successfully!")
            
            # Display Langfuse dashboard link
            st.markdown(f"""
            <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                📊 Open Langfuse Dashboard
            </a>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error initializing Langfuse: {str(e)}")

    # Model selection
    model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    selected_model = st.selectbox("🤖 Select Model", model_options)

    # Model parameters
    st.markdown("### Model Parameters")
    temperature = st.slider("🌡️ Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("📝 Max Output Tokens", 100, 8192, 1024, 100)
    top_p = st.slider("🎯 Top P", 0.0, 1.0, 0.95, 0.05)

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
     <h1>🚀 Advanced Gemini Prompting Studio v2.1</h1>
    <p>Enhanced with Langfuse Tracking & Parallel Execution</p>
</div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🎯 Prompting Techniques",
    "🔍 Chain of Verification", 
    "⚡ Parallel Execution",
    "🔄 Prompt Converter",
    "🧪 Prompt Testing & Evaluation",
    "📊 Analytics Dashboard",
    "📚 Template Library",
    "📥 Export & Import",
    "🧠 Multimodal Prompting"
])

# Prompting technique implementations
class PromptingTechniques:
    def __init__(self, model, langfuse=None):
        self.model = model
        self.langfuse = langfuse

    @observe()
    def zero_shot(self, prompt: str, trace_name: str = "zero_shot") -> str:
        """Zero-shot prompting - direct question without examples"""
        return self._generate_response(prompt, technique="zero_shot", trace_name=trace_name)

    @observe()
    def few_shot(self, task_description: str, examples: List[Dict], query: str, trace_name: str = "few_shot") -> str:
        """Few-shot prompting with examples"""
        prompt = f"{task_description}\n\nExamples:\n"
        for i, example in enumerate(examples, 1):
            prompt += f"{i}. Input: {example['input']} → Output: {example['output']}\n"
        prompt += f"\nNow solve: {query}"
        return self._generate_response(prompt, technique="few_shot", trace_name=trace_name)

    @observe()
    def instruction_style(self, instruction: str, content: str, trace_name: str = "instruction_style") -> str:
        """Direct instruction-based prompting"""
        prompt = f"{instruction}\n\nContent: {content}"
        return self._generate_response(prompt, technique="instruction_style", trace_name=trace_name)

    @observe()
    def chain_of_thought(self, problem: str, trace_name: str = "chain_of_thought") -> str:
        """Chain-of-thought prompting for step-by-step reasoning"""
        prompt = f"{problem}\n\nLet's think step by step:"
        return self._generate_response(prompt, technique="chain_of_thought", trace_name=trace_name)

    @observe()
    def role_based(self, role: str, task: str, trace_name: str = "role_based") -> str:
        """Role-based prompting with specific persona"""
        prompt = f"You are {role}. {task}"
        return self._generate_response(prompt, technique="role_based", trace_name=trace_name)

    @observe()
    def react_prompting(self, task: str, tools_available: List[str], trace_name: str = "react") -> str:
        """ReAct (Reasoning + Acting) prompting"""
        tools_str = ", ".join(tools_available)
        prompt = f"""You have access to the following tools: {tools_str}

Task: {task}

Use the ReAct format:
Thought: [your reasoning about what to do next]
Action: [the action to take]
Observation: [the result of the action]
... (repeat Thought/Action/Observation as needed)
Final Answer: [your final response]

Begin:"""
        return self._generate_response(prompt, technique="react", trace_name=trace_name)

    @observe()
    def chain_of_verification(self, query: str, verification_style: str = "joint", trace_name: str = "chain_of_verification") -> Dict[str, str]:
        """Chain of Verification prompting for improved accuracy"""

        # Step 1: Generate baseline response
        baseline_response = self._generate_response(query, technique="cov_baseline", trace_name=f"{trace_name}_baseline")

        # Step 2: Generate verification questions
        verification_prompt = f"""Given the following question and answer, generate specific verification questions that would help check the accuracy and completeness of the answer. Focus on factual claims, logical consistency, and potential gaps.

Original Question: {query}
Answer: {baseline_response}

Please generate 3-5 specific verification questions that would help validate this answer:"""

        verification_questions = self._generate_response(verification_prompt, technique="cov_questions", trace_name=f"{trace_name}_questions")

        # Step 3: Answer verification questions
        if verification_style == "joint":
            # Joint verification - answer all questions together
            joint_prompt = f"""Please answer the following verification questions based on your knowledge:

{verification_questions}

Provide detailed answers to each question:"""
            verification_answers = self._generate_response(joint_prompt, technique="cov_joint_answers", trace_name=f"{trace_name}_joint_answers")

        else:  # factored verification
            # Factored verification - answer questions independently
            questions_list = [q.strip() for q in verification_questions.split('\n') if q.strip() and ('?' in q)]
            verification_answers = ""
            for i, question in enumerate(questions_list[:5], 1):  # Limit to 5 questions
                answer = self._generate_response(f"Answer this question concisely: {question}", technique="cov_factored_answer", trace_name=f"{trace_name}_factored_{i}")
                verification_answers += f"Q{i}: {question}\nA{i}: {answer}\n\n"

        # Step 4: Generate final verified response
        final_prompt = f"""Based on the original question, initial answer, verification questions, and their answers, provide a final, improved, and verified response. Correct any errors found and incorporate insights from the verification process.

Original Question: {query}
Initial Answer: {baseline_response}

Verification Questions: {verification_questions}
Verification Answers: {verification_answers}

Final Verified Response:"""

        final_response = self._generate_response(final_prompt, technique="cov_final", trace_name=f"{trace_name}_final")

        return {
            "baseline_response": baseline_response,
            "verification_questions": verification_questions,
            "verification_answers": verification_answers,
            "final_response": final_response,
            "verification_style": verification_style
        }

    def parallel_execution(self, prompts: List[Dict[str, Any]], max_workers: int = 3) -> List[Dict[str, Any]]:
        """Execute multiple prompts in parallel"""
        
        def execute_single_prompt(prompt_config):
            try:
                technique = prompt_config.get('technique', 'zero_shot')
                prompt_text = prompt_config.get('prompt', '')
                trace_name = prompt_config.get('trace_name', f"parallel_{technique}")
                
                start_time = time.time()
                
                if technique == 'zero_shot':
                    result = self.zero_shot(prompt_text, trace_name)
                elif technique == 'chain_of_thought':
                    result = self.chain_of_thought(prompt_text, trace_name)
                elif technique == 'role_based':
                    role = prompt_config.get('role', 'expert')
                    result = self.role_based(role, prompt_text, trace_name)
                elif technique == 'few_shot':
                    examples = prompt_config.get('examples', [])
                    task_desc = prompt_config.get('task_description', '')
                    result = self.few_shot(task_desc, examples, prompt_text, trace_name)
                else:
                    result = self.zero_shot(prompt_text, trace_name)
                
                end_time = time.time()
                
                return {
                    'prompt_config': prompt_config,
                    'result': result,
                    'execution_time': end_time - start_time,
                    'status': 'success',
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                return {
                    'prompt_config': prompt_config,
                    'result': f"Error: {str(e)}",
                    'execution_time': 0,
                    'status': 'error',
                    'timestamp': datetime.now()
                }

        # Execute prompts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(execute_single_prompt, prompts))
        
        return results

    def prompt_converter(self, original_prompt: str, source_technique: str, target_technique: str, context: Dict[str, Any] = None) -> str:
        """Convert a prompt from one technique to another"""

        conversion_prompts = {
            "zero_shot_to_few_shot": f"""Convert the following zero-shot prompt into a few-shot prompt with 2-3 examples:

Original prompt: {original_prompt}

Create a few-shot version with:
1. A clear task description
2. 2-3 relevant examples showing input → output format
3. The query to solve

Format as:
Task: [description]
Examples:
1. Input: [example] → Output: [result]
2. Input: [example] → Output: [result]
3. Input: [example] → Output: [result]
Query: [original query]""",

            "zero_shot_to_cot": f"""Convert the following prompt into a Chain-of-Thought prompt that encourages step-by-step reasoning:

Original prompt: {original_prompt}

Add appropriate phrases like 'Let's think step by step', 'First,', 'Then,', etc. to encourage detailed reasoning.""",

            "zero_shot_to_role_based": f"""Convert the following prompt into a role-based prompt by adding an appropriate expert persona:

Original prompt: {original_prompt}

Add a relevant expert role (e.g., 'You are an expert [field]...') that would be most suitable for this task.""",

            "few_shot_to_zero_shot": f"""Convert this few-shot prompt into a clear, direct zero-shot prompt:

Original prompt: {original_prompt}

Extract the core task and create a direct, clear instruction without examples.""",

            "cot_to_zero_shot": f"""Convert this Chain-of-Thought prompt into a direct zero-shot prompt:

Original prompt: {original_prompt}

Remove step-by-step instructions and create a direct, concise question.""",

            "role_based_to_zero_shot": f"""Convert this role-based prompt into a neutral zero-shot prompt:

Original prompt: {original_prompt}

Remove the role specification and create a direct, objective instruction.""",

            "any_to_react": f"""Convert the following prompt into a ReAct (Reasoning + Acting) format:

Original prompt: {original_prompt}

Structure it as:
Task: [clear task description]
Available tools: [relevant tools for this task]

Use ReAct format:
Thought: [reasoning about what to do]
Action: [action to take]
Observation: [expected result]
Final Answer: [conclusion]""",

            "any_to_cov": f"""Convert the following prompt into a Chain-of-Verification prompt:

Original prompt: {original_prompt}

Enhance it to include verification steps that would help validate the accuracy of the response."""
        }

        # Determine conversion key
        conversion_key = f"{source_technique.lower().replace('-', '_')}_to_{target_technique.lower().replace('-', '_')}"

        # Handle generic conversions
        if conversion_key not in conversion_prompts:
            if target_technique.lower().replace('-', '_') == "react":
                conversion_key = "any_to_react"
            elif target_technique.lower().replace('-', '_') == "chain_of_verification":
                conversion_key = "any_to_cov"
            else:
                # Generic conversion prompt
                conversion_prompts[conversion_key] = f"""Convert the following {source_technique} prompt into a {target_technique} prompt:

Original prompt: {original_prompt}

Adapt the prompt structure and format to match the {target_technique} technique while preserving the core intent."""

        prompt_to_convert = conversion_prompts.get(conversion_key, conversion_prompts["any_to_cov"])

        return self._generate_response(prompt_to_convert, technique="prompt_conversion", trace_name="conversion")

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt and suggest improvements"""
        analysis_prompt = f"""Analyze the following prompt and provide detailed feedback:

Prompt: {prompt}

Please provide analysis on:
1. Technique Type: What prompting technique is being used?
2. Clarity: How clear and specific is the prompt?
3. Completeness: What information might be missing?
4. Potential Issues: Any ambiguities or problems?
5. Improvement Suggestions: How could it be enhanced?
6. Estimated Effectiveness: Rate the prompt's likely effectiveness (1-10)

Provide a structured analysis with specific recommendations."""

        analysis_result = self._generate_response(analysis_prompt, technique="analysis", trace_name="prompt_analysis")

        # Extract technique if possible
        technique_detection_prompt = f"""What prompting technique is primarily used in this prompt? Choose from:
- Zero-Shot
- Few-Shot
- Chain-of-Thought
- Role-Based
- Instruction-Style
- ReAct
- Chain-of-Verification
- Mixed/Hybrid

Prompt: {prompt}

Answer with just the technique name:"""

        detected_technique = self._generate_response(technique_detection_prompt, technique="technique_detection", trace_name="technique_detection")

        return {
            "analysis": analysis_result,
            "detected_technique": detected_technique.strip(),
            "prompt": prompt
        }

    def _generate_response(self, prompt: str, technique: str = "general", trace_name: str = "generation") -> str:
        """Helper method to generate response from model with Langfuse tracking"""
        try:
            if not self.model:
                return "❌ Model not initialized. Please provide API key."

            # Create Langfuse trace if available
            if self.langfuse:
                trace = self.langfuse.trace(
                    name=trace_name,
                    metadata={
                        "technique": technique,
                        "model": self.model.model_name if hasattr(self.model, 'model_name') else "gemini",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Create generation span
                generation = trace.generation(
                    name=f"{technique}_generation",
                    model=self.model.model_name if hasattr(self.model, 'model_name') else "gemini",
                    model_parameters={
                        "temperature": self.model._generation_config.temperature if hasattr(self.model, '_generation_config') else None,
                        "max_output_tokens": self.model._generation_config.max_output_tokens if hasattr(self.model, '_generation_config') else None,
                    },
                    input=prompt
                )

            response = self.model.generate_content(prompt)
            response_text = response.text

            # Update Langfuse generation with output
            if self.langfuse and 'generation' in locals():
                generation.end(output=response_text)

            return response_text
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            
            # Log error to Langfuse if available
            if self.langfuse and 'generation' in locals():
                generation.end(output=error_msg, level="ERROR")
            
            return error_msg

# Template library (updated with CoV and Parallel templates)
TEMPLATE_LIBRARY = {
    "Zero-Shot": {
        "Translation": "Translate the following text to {target_language}: '{text}'",
        "Summarization": "Summarize the following text in {num_sentences} sentences: '{text}'",
        "Question Answering": "Answer the following question: {question}",
        "Code Generation": "Write {language} code to {task}",
        "Creative Writing": "Write a {genre} story about {topic}",
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
        }
    },
    "Chain-of-Thought": {
        "Math Problem": "A train travels at {speed} km/h for {time} hours. How far does it travel?",
        "Logic Puzzle": "If all cats are animals, and some animals are pets, can we conclude that some cats are pets?",
        "Word Problem": "Sarah has {initial_amount} apples. She gives {given_amount} to her friend. How many does she have left?"
    },
    "Role-Based": {
        "Expert Consultant": "You are an expert {field} consultant. Provide advice on {topic}.",
        "Teacher": "You are a {subject} teacher explaining {concept} to {grade_level} students.",
        "Creative Writer": "You are a creative writer in the {genre} genre. Write about {theme}.",
        "Technical Reviewer": "You are a senior software engineer reviewing code. Analyze this {language} code: {code}"
    },
    "ReAct": {
        "Research Task": "Research and provide comprehensive information about {topic}. Use web search, analysis, and synthesis.",
        "Problem Solving": "Solve this complex problem step by step: {problem}. Use logical reasoning and available tools.",
        "Decision Making": "Help me make a decision about {decision_context}. Consider pros, cons, and alternatives."
    },
    "Chain-of-Verification": {
        "Factual Claims": "What are the main causes of {phenomenon}? Provide a comprehensive and verified answer.",
        "Historical Events": "Describe the key events and significance of {historical_event}.",
        "Scientific Explanations": "Explain how {scientific_concept} works and its applications.",
        "Complex Reasoning": "Analyze the relationship between {concept_a} and {concept_b} and provide evidence-based conclusions."
    },
    "Parallel": {
        "Comparative Analysis": "Compare {topic_a} and {topic_b} from different perspectives",
        "Multi-Perspective": "Analyze {topic} from the viewpoints of {perspective_1}, {perspective_2}, and {perspective_3}",
        "A/B Testing": "Create two different approaches to solve: {problem}",
        "Brainstorming": "Generate multiple creative solutions for: {challenge}"
    }
}

# Tab 1: Prompting Techniques
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🎯 Select Technique")
        technique = st.selectbox(
            "Choose prompting technique:",
            ["Zero-Shot", "Few-Shot", "Instruction-Style", "Chain-of-Thought", "Role-Based", "ReAct"]
        )

        # Dynamic input based on technique
        if technique == "Zero-Shot":
            prompt_input = st.text_area("💭 Enter your prompt:", height=100)

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

        elif technique == "Instruction-Style":
            instruction = st.text_input("📋 Instruction:", placeholder="e.g., Summarize, Translate, Explain...")
            content = st.text_area("📄 Content:", height=100)

        elif technique == "Chain-of-Thought":
            prompt_input = st.text_area("🧠 Problem to solve:", height=100)

        elif technique == "Role-Based":
            role = st.text_input("👤 Role/Persona:", placeholder="e.g., expert data scientist, creative writer...")
            task = st.text_area("📋 Task:", height=100)

        elif technique == "ReAct":
            task_input = st.text_area("🎯 Task:", height=100)
            tools = st.multiselect(
                "🛠️ Available Tools:",
                ["Web Search", "Calculator", "Code Executor", "Knowledge Base", "Data Analysis"]
            )

        # Langfuse tracking option
        track_langfuse = st.checkbox("📊 Track with Langfuse", value=True, help="Enable Langfuse tracking for this request")

    with col2:
        st.markdown("### 🚀 Results")

        if st.button("🎯 Generate Response", type="primary"):
            if not st.session_state.model:
                st.error("❌ Please configure your API key first!")
            else:
                with st.spinner("🤖 Generating response..."):
                    techniques = PromptingTechniques(
                        st.session_state.model, 
                        st.session_state.langfuse if track_langfuse else None
                    )

                    # Generate response based on technique
                    if technique == "Zero-Shot" and prompt_input:
                        response = techniques.zero_shot(prompt_input, f"single_{technique.lower()}")
                        used_prompt = prompt_input

                    elif technique == "Few-Shot" and task_desc and examples and query:
                        response = techniques.few_shot(task_desc, examples, query, f"single_{technique.lower()}")
                        used_prompt = f"Task: {task_desc}, Examples: {examples}, Query: {query}"

                    elif technique == "Instruction-Style" and instruction and content:
                        response = techniques.instruction_style(instruction, content, f"single_{technique.lower()}")
                        used_prompt = f"Instruction: {instruction}, Content: {content}"

                    elif technique == "Chain-of-Thought" and prompt_input:
                        response = techniques.chain_of_thought(prompt_input, f"single_{technique.lower()}")
                        used_prompt = prompt_input

                    elif technique == "Role-Based" and role and task:
                        response = techniques.role_based(role, task, f"single_{technique.lower()}")
                        used_prompt = f"Role: {role}, Task: {task}"

                    elif technique == "ReAct" and task_input and tools:
                        response = techniques.react_prompting(task_input, tools, f"single_{technique.lower()}")
                        used_prompt = f"Task: {task_input}, Tools: {tools}"

                    else:
                        st.warning("⚠️ Please fill in all required fields!")
                        response = None
                        used_prompt = None

                    if response and used_prompt:
                        # Display response
                        st.markdown(f"""
                        <div class="prompt-result">
                            <h4>🤖 Response:</h4>
                            <p>{response}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Display Langfuse link if tracking enabled
                        if track_langfuse and st.session_state.langfuse_url:
                            st.markdown(f"""
                            <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                                📊 View in Langfuse Dashboard
                            </a>
                            """, unsafe_allow_html=True)

                        # Save to history
                        st.session_state.prompts_history.append({
                            "timestamp": datetime.now(),
                            "technique": technique,
                            "prompt": used_prompt,
                            "response": response,
                            "model": selected_model,
                            "parameters": {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p},
                            "langfuse_tracked": track_langfuse
                        })

                        st.success("✅ Response saved to history!")

# Tab 2: Chain of Verification (existing implementation with langfuse tracking)
with tab2:
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

        # Langfuse tracking
        track_cov_langfuse = st.checkbox("📊 Track with Langfuse", value=True, key="cov_langfuse")

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
            st.text_area("Template Query:", value=cov_query, height=80, disabled=True, key="template_display")

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
                        st.session_state.langfuse if track_cov_langfuse else None
                    )

                    # Run CoV process
                    cov_result = techniques.chain_of_verification(cov_query, verification_style, "cov_complete")

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

                    # Display Langfuse link if tracking enabled
                    if track_cov_langfuse and st.session_state.langfuse_url:
                        st.markdown(f"""
                        <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                            📊 View CoV Process in Langfuse
                        </a>
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
                            "baseline_response": cov_result['baseline_response'],
                            "verification_questions": cov_result['verification_questions'],
                            "verification_answers": cov_result['verification_answers']
                        },
                        "langfuse_tracked": track_cov_langfuse
                    })

                    st.success("✅ Chain of Verification completed and saved to history!")

# Tab 3: Parallel Execution (NEW)
with tab3:
    st.markdown("### ⚡ Parallel Prompt Execution")

    st.info("""
    **Parallel Execution** allows you to:
    🚀 Run multiple prompts simultaneously for faster processing
    📊 Compare different approaches to the same problem
    🔄 Test variations of prompts efficiently
    💡 Generate diverse perspectives on a topic
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### ⚙️ Configuration")

        # Execution settings
        max_workers = st.slider("🔧 Max Parallel Workers", 1, 10, 3, help="Number of prompts to execute simultaneously")
        track_parallel_langfuse = st.checkbox("📊 Track with Langfuse", value=True, key="parallel_langfuse")

        # Prompt input method
        input_method = st.radio("📝 Input Method:", ["Manual Entry", "Template-based", "Upload CSV"])

        if input_method == "Manual Entry":
            st.markdown("#### 📝 Add Prompts Manually")
            
            # Initialize session state for prompts
            if 'parallel_prompts' not in st.session_state:
                st.session_state.parallel_prompts = []

            # Add new prompt form
            with st.expander("➕ Add New Prompt"):
                new_technique = st.selectbox("Technique:", ["zero_shot", "chain_of_thought", "role_based", "few_shot"], key="new_technique")
                new_prompt = st.text_area("Prompt:", key="new_prompt")
                
                if new_technique == "role_based":
                    new_role = st.text_input("Role:", key="new_role", value="expert")
                else:
                    new_role = "expert"

                if new_technique == "few_shot":
                    new_task_desc = st.text_input("Task Description:", key="new_task_desc")
                    new_examples = st.text_area("Examples (JSON format):", key="new_examples", 
                                               value='[{"input": "example", "output": "result"}]')
                else:
                    new_task_desc = ""
                    new_examples = "[]"

                new_trace_name = st.text_input("Trace Name:", key="new_trace", value=f"parallel_{len(st.session_state.parallel_prompts)+1}")

                if st.button("➕ Add Prompt"):
                    try:
                        examples_list = json.loads(new_examples) if new_examples else []
                        prompt_config = {
                            "technique": new_technique,
                            "prompt": new_prompt,
                            "role": new_role,
                            "task_description": new_task_desc,
                            "examples": examples_list,
                            "trace_name": new_trace_name
                        }
                        st.session_state.parallel_prompts.append(prompt_config)
                        st.success(f"✅ Added prompt {len(st.session_state.parallel_prompts)}")
                        st.experimental_rerun()
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON format for examples")

            # Display current prompts
            if st.session_state.parallel_prompts:
                st.markdown("#### 📋 Current Prompts")
                for i, prompt_config in enumerate(st.session_state.parallel_prompts):
                    with st.expander(f"Prompt {i+1}: {prompt_config['technique']}"):
                        st.write(f"**Technique:** {prompt_config['technique']}")
                        st.write(f"**Prompt:** {prompt_config['prompt'][:100]}...")
                        if prompt_config['technique'] == 'role_based':
                            st.write(f"**Role:** {prompt_config['role']}")
                        if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                            st.session_state.parallel_prompts.pop(i)
                            st.experimental_rerun()

        elif input_method == "Template-based":
            st.markdown("#### 📚 Template-based Prompts")
            
            template_category = st.selectbox("Template Category:", ["Parallel"] + list(TEMPLATE_LIBRARY.keys()))
            
            if template_category == "Parallel":
                parallel_template = st.selectbox("Parallel Template:", list(TEMPLATE_LIBRARY["Parallel"].keys()))
                template_text = TEMPLATE_LIBRARY["Parallel"][parallel_template]
                
                # Extract variables
                variables = re.findall(r'\{(\w+)\}', template_text)
                var_values = {}
                
                for var in variables:
                    var_values[var] = st.text_input(f"{var.replace('_', ' ').title()}:", key=f"parallel_var_{var}")
                
                if st.button("🎯 Generate Parallel Prompts"):
                    if all(var_values.values()):
                        filled_template = template_text.format(**var_values)
                        
                        # Generate different variations based on template type
                        if parallel_template == "Comparative Analysis":
                            prompts = [
                                {"technique": "zero_shot", "prompt": f"Analyze {var_values.get('topic_a', '')}: {filled_template}", "trace_name": f"compare_a"},
                                {"technique": "zero_shot", "prompt": f"Analyze {var_values.get('topic_b', '')}: {filled_template}", "trace_name": f"compare_b"},
                                {"technique": "chain_of_thought", "prompt": f"Step by step comparison: {filled_template}", "trace_name": f"compare_cot"}
                            ]
                        elif parallel_template == "Multi-Perspective":
                            perspectives = [var_values.get('perspective_1', ''), var_values.get('perspective_2', ''), var_values.get('perspective_3', '')]
                            prompts = [
                                {"technique": "role_based", "role": f"expert {perspective}", "prompt": filled_template, "trace_name": f"perspective_{i+1}"}
                                for i, perspective in enumerate(perspectives) if perspective
                            ]
                        else:
                            # Default: create multiple zero-shot variations
                            prompts = [
                                {"technique": "zero_shot", "prompt": filled_template, "trace_name": f"template_{i+1}"}
                                for i in range(3)
                            ]
                        
                        st.session_state.parallel_prompts = prompts
                        st.success(f"✅ Generated {len(prompts)} prompts from template")
                    else:
                        st.warning("⚠️ Please fill all template variables")

        elif input_method == "Upload CSV":
            st.markdown("#### 📁 Upload CSV File")
            
            uploaded_file = st.file_uploader("Upload CSV with prompts", type=['csv'])
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head())
                    
                    required_columns = ['technique', 'prompt']
                    if all(col in df.columns for col in required_columns):
                        prompts = []
                        for _, row in df.iterrows():
                            prompt_config = {
                                "technique": row.get('technique', 'zero_shot'),
                                "prompt": row.get('prompt', ''),
                                "role": row.get('role', 'expert'),
                                "task_description": row.get('task_description', ''),
                                "examples": json.loads(row.get('examples', '[]')) if pd.notna(row.get('examples')) else [],
                                "trace_name": row.get('trace_name', f"csv_{len(prompts)+1}")
                            }
                            prompts.append(prompt_config)
                        
                        st.session_state.parallel_prompts = prompts
                        st.success(f"✅ Loaded {len(prompts)} prompts from CSV")
                    else:
                        st.error(f"❌ CSV must contain columns: {required_columns}")
                        
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")

        # Clear all prompts button
        if st.session_state.get('parallel_prompts'):
            if st.button("🗑️ Clear All Prompts"):
                st.session_state.parallel_prompts = []
                st.experimental_rerun()

    with col2:
        st.markdown("#### 🚀 Execution Results")

        if st.button("⚡ Execute Parallel Prompts", type="primary"):
            if not st.session_state.get('parallel_prompts'):
                st.warning("⚠️ Please add some prompts first!")
            elif not st.session_state.model:
                st.error("❌ Please configure your API key first!")
            else:
                with st.spinner("⚡ Executing prompts in parallel..."):
                    techniques = PromptingTechniques(
                        st.session_state.model,
                        st.session_state.langfuse if track_parallel_langfuse else None
                    )

                    # Execute prompts in parallel
                    start_time = time.time()
                    results = techniques.parallel_execution(st.session_state.parallel_prompts, max_workers)
                    total_time = time.time() - start_time

                    # Display execution summary
                    success_count = sum(1 for r in results if r['status'] == 'success')
                    error_count = len(results) - success_count
                    avg_time = sum(r['execution_time'] for r in results) / len(results) if results else 0

                    col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
                    
                    with col_summary1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{len(results)}</h3>
                            <p>Total Prompts</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_summary2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{success_count}</h3>
                            <p>Successful</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_summary3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{total_time:.2f}s</h3>
                            <p>Total Time</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_summary4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{avg_time:.2f}s</h3>
                            <p>Avg per Prompt</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Display individual results
                    st.markdown("#### 📋 Individual Results")
                    
                    for i, result in enumerate(results):
                        status_icon = "✅" if result['status'] == 'success' else "❌"
                        
                        with st.expander(f"{status_icon} Result {i+1}: {result['prompt_config']['technique']} ({result['execution_time']:.2f}s)"):
                            col_config, col_result = st.columns([1, 2])
                            
                            with col_config:
                                st.markdown("**Configuration:**")
                                st.json(result['prompt_config'])
                            
                            with col_result:
                                st.markdown("**Result:**")
                                if result['status'] == 'success':
                                    st.markdown(f"""
                                    <div class="parallel-result">
                                        <p>{result['result']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="error-message">
                                        <p>{result['result']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                    # Performance chart
                    if len(results) > 1:
                        st.markdown("#### 📊 Performance Analysis")
                        
                        # Create performance dataframe
                        perf_data = []
                        for i, result in enumerate(results):
                            perf_data.append({
                                'Prompt': f"Prompt {i+1}",
                                'Technique': result['prompt_config']['technique'],
                                'Execution Time (s)': result['execution_time'],
                                'Status': result['status'],
                                'Response Length': len(result['result']) if result['status'] == 'success' else 0
                            })
                        
                        perf_df = pd.DataFrame(perf_data)
                        
                        # Execution time chart
                        fig_time = px.bar(perf_df, x='Prompt', y='Execution Time (s)', 
                                         color='Technique', title="Execution Time by Prompt")
                        st.plotly_chart(fig_time, use_container_width=True)
                        
                        # Response length chart
                        if any(perf_df['Response Length'] > 0):
                            fig_length = px.scatter(perf_df, x='Execution Time (s)', y='Response Length',
                                                  color='Technique', title="Response Length vs Execution Time")
                            st.plotly_chart(fig_length, use_container_width=True)

                    # Display Langfuse link if tracking enabled
                    if track_parallel_langfuse and st.session_state.langfuse_url:
                        st.markdown(f"""
                        <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                            📊 View Parallel Execution in Langfuse
                        </a>
                        """, unsafe_allow_html=True)

                    # Save to history
                    for result in results:
                        if result['status'] == 'success':
                            st.session_state.prompts_history.append({
                                "timestamp": result['timestamp'],
                                "technique": f"Parallel-{result['prompt_config']['technique']}",
                                "prompt": result['prompt_config']['prompt'],
                                "response": result['result'],
                                "model": selected_model,
                                "parameters": {
                                    "temperature": temperature,
                                    "max_tokens": max_tokens,
                                    "top_p": top_p,
                                    "execution_time": result['execution_time'],
                                    "parallel_execution": True,
                                    "max_workers": max_workers
                                },
                                "langfuse_tracked": track_parallel_langfuse
                            })

                    st.success(f"✅ Parallel execution completed! {success_count}/{len(results)} prompts successful")

# Tab 4: Prompt Converter (existing implementation with minimal changes)
with tab4:
    st.markdown("### 🔄 Prompt Converter & Analyzer")

    st.info("""
    **Prompt Converter** helps you:
    🔄 Convert prompts between different techniques
    📊 Analyze prompt quality and effectiveness
    💡 Get suggestions for improvement
    🎯 Optimize prompts for better results
    """)

    converter_tab1, converter_tab2 = st.tabs(["🔄 Convert Prompts", "📊 Analyze Prompts"])

    with converter_tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### 🎯 Conversion Settings")

            # Input prompt
            input_prompt = st.text_area("📝 Enter your prompt:",
                                      placeholder="Enter the prompt you want to convert...",
                                      height=120)

            # Technique selection
            technique_options = [
                "Zero-Shot", "Few-Shot", "Chain-of-Thought",
                "Role-Based", "Instruction-Style", "ReAct", "Chain-of-Verification"
            ]

            col_source, col_target = st.columns(2)

            with col_source:
                source_technique = st.selectbox("📥 From Technique:",
                                              ["Auto-Detect"] + technique_options)

            with col_target:
                target_technique = st.selectbox("📤 To Technique:", technique_options)

            # Additional context
            st.markdown("#### ⚙️ Conversion Context")
            domain = st.text_input("🏷️ Domain/Subject:",
                                 placeholder="e.g., mathematics, creative writing, data analysis")

            complexity = st.select_slider("🎚️ Target Complexity:",
                                        options=["Simple", "Moderate", "Complex", "Expert"],
                                        value="Moderate")

            audience = st.text_input("👥 Target Audience:",
                                   placeholder="e.g., beginners, professionals, students")

            track_converter_langfuse = st.checkbox("📊 Track with Langfuse", value=True, key="converter_langfuse")

        with col2:
            st.markdown("#### 🚀 Conversion Results")

            if st.button("🔄 Convert Prompt", type="primary"):
                if not input_prompt:
                    st.warning("⚠️ Please enter a prompt to convert!")
                elif not st.session_state.model:
                    st.error("❌ Please configure your API key first!")
                else:
                    with st.spinner("🔄 Converting prompt..."):
                        techniques = PromptingTechniques(
                            st.session_state.model,
                            st.session_state.langfuse if track_converter_langfuse else None
                        )

                        # Auto-detect source technique if needed
                        if source_technique == "Auto-Detect":
                            analysis = techniques.analyze_prompt(input_prompt)
                            detected_technique = analysis['detected_technique']
                            st.info(f"🔍 Detected technique: **{detected_technique}**")
                            source_tech = detected_technique
                        else:
                            source_tech = source_technique

                        # Prepare context
                        context = {
                            "domain": domain,
                            "complexity": complexity,
                            "audience": audience
                        }

                        # Convert prompt
                        converted_prompt = techniques.prompt_converter(
                            input_prompt, source_tech, target_technique, context
                        )

                        # Display results
                        st.markdown("##### 📥 Original Prompt")
                        st.markdown(f"""
                        <div class="verification-step">
                            <strong>Technique:</strong> {source_tech}<br>
                            <strong>Prompt:</strong> {input_prompt}
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("##### 📤 Converted Prompt")
                        st.markdown(f"""
                        <div class="prompt-result">
                            <strong>Technique:</strong> {target_technique}<br><br>
                            {converted_prompt}
                        </div>
                        """, unsafe_allow_html=True)

                        # Display Langfuse link if tracking enabled
                        if track_converter_langfuse and st.session_state.langfuse_url:
                            st.markdown(f"""
                            <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                                📊 View Conversion in Langfuse
                            </a>
                            """, unsafe_allow_html=True)

                        # Test both prompts option
                        if st.button("⚡ Test Both Prompts"):
                            st.markdown("##### 🧪 Comparison Results")

                            col_orig, col_conv = st.columns(2)

                            with col_orig:
                                st.markdown("**Original Response:**")
                                orig_response = techniques.zero_shot(input_prompt, "original_test")
                                st.write(orig_response)

                            with col_conv:
                                st.markdown("**Converted Response:**")
                                conv_response = techniques.zero_shot(converted_prompt, "converted_test")
                                st.write(conv_response)

                        # Save to history
                        st.session_state.prompts_history.append({
                            "timestamp": datetime.now(),
                            "technique": f"Conversion: {source_tech} → {target_technique}",
                            "prompt": converted_prompt,
                            "response": f"Converted from: {input_prompt}",
                            "model": selected_model,
                            "parameters": {
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "top_p": top_p,
                                "conversion_context": context,
                                "source_technique": source_tech,
                                "target_technique": target_technique
                            },
                            "langfuse_tracked": track_converter_langfuse
                        })

                        st.success("✅ Prompt converted and saved to history!")

    with converter_tab2:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### 📊 Prompt Analysis")

            # Input prompt for analysis
            analysis_prompt = st.text_area("📝 Enter prompt to analyze:",
                                         placeholder="Enter any prompt for detailed analysis...",
                                         height=120, key="analysis_input")

            # Analysis options
            analysis_depth = st.radio("🔍 Analysis Depth:",
                                    ["Quick", "Detailed", "Comprehensive"])

            focus_areas = st.multiselect("🎯 Focus Areas:",
                                       ["Clarity", "Specificity", "Structure", "Effectiveness",
                                        "Potential Issues", "Improvement Suggestions"],
                                       default=["Clarity", "Effectiveness"])

            track_analysis_langfuse = st.checkbox("📊 Track with Langfuse", value=True, key="analysis_langfuse")

        with col2:
            st.markdown("#### 📈 Analysis Results")

            if st.button("📊 Analyze Prompt", type="primary"):
                if not analysis_prompt:
                    st.warning("⚠️ Please enter a prompt to analyze!")
                elif not st.session_state.model:
                    st.error("❌ Please configure your API key first!")
                else:
                    with st.spinner("📊 Analyzing prompt..."):
                        techniques = PromptingTechniques(
                            st.session_state.model,
                            st.session_state.langfuse if track_analysis_langfuse else None
                        )

                        # Perform analysis
                        analysis_result = techniques.analyze_prompt(analysis_prompt)

                        # Display detected technique
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🔍 Detected Technique</h4>
                            <h2>{analysis_result['detected_technique']}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                        # Display full analysis
                        st.markdown("##### 📋 Detailed Analysis")
                        st.markdown(f"""
                        <div class="prompt-result">
                            {analysis_result['analysis']}
                        </div>
                        """, unsafe_allow_html=True)

                        # Display Langfuse link if tracking enabled
                        if track_analysis_langfuse and st.session_state.langfuse_url:
                            st.markdown(f"""
                            <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                                📊 View Analysis in Langfuse
                            </a>
                            """, unsafe_allow_html=True)

                        # Suggest alternative techniques
                        st.markdown("##### 💡 Alternative Techniques")

                        alternative_techniques = [t for t in technique_options
                                                if t != analysis_result['detected_technique']]

                        selected_alternatives = st.multiselect(
                            "Convert to these techniques for comparison:",
                            alternative_techniques,
                            max_selections=3
                        )

                        if selected_alternatives:
                            for alt_technique in selected_alternatives:
                                with st.expander(f"🔄 Convert to {alt_technique}"):
                                    converted = techniques.prompt_converter(
                                        analysis_prompt,
                                        analysis_result['detected_technique'],
                                        alt_technique
                                    )
                                    st.code(converted, language="text")

                        # Save analysis to history
                        st.session_state.prompts_history.append({
                            "timestamp": datetime.now(),
                            "technique": "Prompt Analysis",
                            "prompt": analysis_prompt,
                            "response": analysis_result['analysis'],
                            "model": selected_model,
                            "parameters": {
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "top_p": top_p,
                                "detected_technique": analysis_result['detected_technique'],
                                "analysis_depth": analysis_depth,
                                "focus_areas": focus_areas
                            },
                            "langfuse_tracked": track_analysis_langfuse
                        })

                        st.success("✅ Analysis completed and saved to history!")

# Tab 5: Prompt Testing & Evaluation (existing implementation)
with tab5:
    st.markdown("### 🧪 Prompt Testing & Evaluation")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📝 Batch Testing")

        # Upload test cases
        uploaded_file = st.file_uploader("📁 Upload test cases (CSV)", type=['csv'])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            track_testing_langfuse = st.checkbox("📊 Track with Langfuse", value=True, key="testing_langfuse")

            if st.button("🧪 Run Batch Tests"):
                if not st.session_state.model:
                    st.error("❌ Please configure your API key first!")
                else:
                    results = []
                    progress_bar = st.progress(0)

                    techniques = PromptingTechniques(
                        st.session_state.model,
                        st.session_state.langfuse if track_testing_langfuse else None
                    )

                    for idx, row in df.iterrows():
                        prompt = row.get('prompt', '')
                        expected = row.get('expected', '')

                        if prompt:
                            response = techniques.zero_shot(prompt, f"batch_test_{idx}")

                            results.append({
                                'prompt': prompt,
                                'expected': expected,
                                'actual': response,
                                'timestamp': datetime.now()
                            })

                        progress_bar.progress((idx + 1) / len(df))

                    st.session_state.evaluation_results = results

                    # Display Langfuse link if tracking enabled
                    if track_testing_langfuse and st.session_state.langfuse_url:
                        st.markdown(f"""
                        <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                            📊 View Batch Tests in Langfuse
                        </a>
                        """, unsafe_allow_html=True)

                    st.success(f"✅ Completed {len(results)} tests!")

        # Manual testing
        st.markdown("#### ✋ Manual Testing")
        test_prompt = st.text_area("🧪 Test Prompt:", height=100)
        expected_output = st.text_area("🎯 Expected Output:", height=100)

        track_manual_langfuse = st.checkbox("📊 Track with Langfuse", value=True, key="manual_langfuse")

        if st.button("🧪 Run Single Test"):
            if test_prompt and st.session_state.model:
                techniques = PromptingTechniques(
                    st.session_state.model,
                    st.session_state.langfuse if track_manual_langfuse else None
                )
                actual_output = techniques.zero_shot(test_prompt, "manual_test")

                st.markdown("**Results:**")
                col_exp, col_act = st.columns(2)

                with col_exp:
                    st.markdown("**Expected:**")
                    st.write(expected_output)

                with col_act:
                    st.markdown("**Actual:**")
                    st.write(actual_output)

                # Simple similarity score
                if expected_output and actual_output:
                    similarity = len(set(expected_output.lower().split()) & set(actual_output.lower().split())) / max(len(set(expected_output.lower().split())), len(set(actual_output.lower().split())), 1)
                    st.metric("📊 Word Overlap Score", f"{similarity:.2%}")

                # Display Langfuse link if tracking enabled
                if track_manual_langfuse and st.session_state.langfuse_url:
                    st.markdown(f"""
                    <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                        📊 View Test in Langfuse
                    </a>
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📊 Evaluation Metrics")

        if st.session_state.evaluation_results:
            results_df = pd.DataFrame(st.session_state.evaluation_results)

            # Calculate metrics
            total_tests = len(results_df)
            avg_response_length = results_df['actual'].str.len().mean()

            # Display metrics
            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_tests}</h3>
                    <p>Total Tests</p>
                </div>
                """, unsafe_allow_html=True)

            with metric_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{avg_response_length:.0f}</h3>
                    <p>Avg Response Length</p>
                </div>
                """, unsafe_allow_html=True)

            # Response length distribution
            fig = px.histogram(results_df, x=results_df['actual'].str.len(),
                             title="Response Length Distribution")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("📊 Run some tests to see evaluation metrics!")

# Tab 6: Analytics Dashboard (enhanced with Langfuse tracking info)
with tab6:
    st.markdown("### 📊 Analytics Dashboard")

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
            avg_response_length = df_history['response'].str.len().mean()
            st.metric("📝 Avg Response Length", f"{avg_response_length:.0f}")

        with col4:
            langfuse_tracked = df_history.get('langfuse_tracked', pd.Series([False] * len(df_history))).sum()
            st.metric("📊 Langfuse Tracked", langfuse_tracked)

        # Langfuse dashboard link
        if st.session_state.langfuse_url:
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                    📊 Open Full Langfuse Analytics Dashboard
                </a>
            </div>
            """, unsafe_allow_html=True)

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Technique usage
            technique_counts = df_history['technique'].value_counts()
            fig_pie = px.pie(values=technique_counts.values, names=technique_counts.index,
                           title="Prompting Techniques Usage")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Usage over time
            df_history['date'] = pd.to_datetime(df_history['timestamp']).dt.date
            daily_usage = df_history.groupby('date').size().reset_index(name='count')
            fig_line = px.line(daily_usage, x='date', y='count',
                             title="Daily Usage Trend")
            st.plotly_chart(fig_line, use_container_width=True)

        # Langfuse tracking analysis
        if 'langfuse_tracked' in df_history.columns:
            langfuse_usage = df_history.groupby(['technique', 'langfuse_tracked']).size().reset_index(name='count')
            fig_langfuse = px.bar(langfuse_usage, x='technique', y='count', color='langfuse_tracked',
                                title="Langfuse Tracking by Technique")
            fig_langfuse.update_xaxes(tickangle=45)
            st.plotly_chart(fig_langfuse, use_container_width=True)

        # Response length by technique
        fig_box = px.box(df_history, x='technique', y=df_history['response'].str.len(),
                        title="Response Length by Technique")
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

        # Recent prompts table
        st.markdown("#### 📝 Recent Prompts")
        recent_prompts = df_history.tail(10)[['timestamp', 'technique', 'prompt', 'model', 'langfuse_tracked']].copy()
        recent_prompts['prompt'] = recent_prompts['prompt'].str[:100] + '...'
        st.dataframe(recent_prompts, use_container_width=True)

    else:
        st.info("📊 No data available yet. Start using prompting techniques to see analytics!")

# Tab 7: Template Library (existing implementation with parallel templates)
with tab7:
    st.markdown("### 📚 Template Library")

    # Select category
    category = st.selectbox("📂 Select Category:", list(TEMPLATE_LIBRARY.keys()))

    if category in TEMPLATE_LIBRARY:
        templates = TEMPLATE_LIBRARY[category]

        col1, col2 = st.columns([1, 2])

        with col1:
            template_name = st.selectbox("📄 Select Template:", list(templates.keys()))

            if template_name in templates:
                template = templates[template_name]

                if isinstance(template, str):
                    # Simple string template
                    st.markdown("**Template:**")
                    st.code(template, language="text")

                    # Extract variables from template
                    variables = re.findall(r'\{(\w+)\}', template)

                    if variables:
                        st.markdown("**Fill Variables:**")
                        var_values = {}
                        for var in variables:
                            var_values[var] = st.text_input(f"{var}:", key=f"template_var_{var}")

                        if st.button("📝 Generate Prompt"):
                            filled_template = template.format(**var_values)
                            st.session_state.filled_template = filled_template

                elif isinstance(template, dict):
                    # Complex template with examples
                    st.markdown("**Description:**")
                    st.write(template.get('description', ''))

                    if 'examples' in template:
                        st.markdown("**Examples:**")
                        for i, example in enumerate(template['examples'], 1):
                            st.write(f"{i}. Input: {example['input']} → Output: {example['output']}")

            track_template_langfuse = st.checkbox("📊 Track with Langfuse", value=True, key="template_langfuse")

        with col2:
            st.markdown("### 🎯 Generated Prompt")

            if hasattr(st.session_state, 'filled_template'):
                st.markdown("**Ready to use:**")
                st.code(st.session_state.filled_template, language="text")

                if st.button("🚀 Execute Prompt"):
                    if st.session_state.model:
                        techniques = PromptingTechniques(
                            st.session_state.model,
                            st.session_state.langfuse if track_template_langfuse else None
                        )
                        response = techniques.zero_shot(st.session_state.filled_template, f"template_{template_name}")

                        st.markdown("**Response:**")
                        st.markdown(f"""
                        <div class="prompt-result">
                            <p>{response}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Display Langfuse link if tracking enabled
                        if track_template_langfuse and st.session_state.langfuse_url:
                            st.markdown(f"""
                            <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                                📊 View Template Execution in Langfuse
                            </a>
                            """, unsafe_allow_html=True)
            else:
                st.info("👆 Select a template and fill variables to generate prompt")

# Tab 8: Export & Import (existing implementation)
with tab8:
    st.markdown("### 📥 Export & Import")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📤 Export Data")

        export_type = st.selectbox("📁 Export Format:", ["JSON", "CSV", "TXT", "PDF"])

        if st.session_state.prompts_history:
            if st.button("📥 Download History"):
                df_export = pd.DataFrame(st.session_state.prompts_history)

                if export_type == "CSV":
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"prompts_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                elif export_type == "JSON":
                    json_data = df_export.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"prompts_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                elif export_type == "TXT":
                    txt_content = ""
                    for item in st.session_state.prompts_history:
                        txt_content += f"Timestamp: {item['timestamp']}\n"
                        txt_content += f"Technique: {item['technique']}\n"
                        txt_content += f"Prompt: {item['prompt']}\n"
                        txt_content += f"Response: {item['response']}\n"
                        txt_content += f"Langfuse Tracked: {item.get('langfuse_tracked', False)}\n"
                        txt_content += "-" * 50 + "\n\n"

                    st.download_button(
                        label="📥 Download TXT",
                        data=txt_content,
                        file_name=f"prompts_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("📊 No history data to export!")

    with col2:
        st.markdown("#### 📤 Import Data")

        uploaded_history = st.file_uploader("📁 Upload History File", type=['json', 'csv'])

        if uploaded_history:
            try:
                if uploaded_history.name.endswith('.json'):
                    imported_data = json.load(uploaded_history)
                    st.session_state.prompts_history.extend(imported_data)
                    st.success(f"✅ Imported {len(imported_data)} records!")

                elif uploaded_history.name.endswith('.csv'):
                    df_imported = pd.read_csv(uploaded_history)
                    imported_data = df_imported.to_dict('records')
                    st.session_state.prompts_history.extend(imported_data)
                    st.success(f"✅ Imported {len(imported_data)} records!")

            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

        # Clear history
        st.markdown("#### 🗑️ Data Management")
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.prompts_history = []
            st.session_state.evaluation_results = []
            st.success("✅ History cleared!")

# Tab 9: Multimodal Prompting (existing implementation with langfuse tracking)
with tab9:
    st.markdown("### 🧠 Multimodal Prompting")

    if not st.session_state.model:
        st.warning("⚠️ Please configure your Gemini API key in the sidebar.")
    else:
        st.info("📸 Upload images and provide text or voice prompts for multimodal AI interactions")

        col1, col2 = st.columns([1, 2])

        with col1:
            # File uploads
            image_files = st.file_uploader("📸 Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

            # Prompt input method
            prompt_method = st.radio("🎤 Prompt Input Method:", ["📝 Text Input", "🗣️ Voice Note"])

            multimodal_prompt = ""

            if prompt_method == "📝 Text Input":
                multimodal_prompt = st.text_area("💭 Enter your prompt:", height=100, key="multimodal_text_prompt")

            elif prompt_method == "🗣️ Voice Note":
                st.info("🎙️ Record a voice note for your prompt")
                # Note: In a real implementation, you'd integrate with speech recognition
                multimodal_prompt = st.text_area("🎤 Voice prompt will appear here:",
                                               placeholder="Voice transcription will appear here...",
                                               height=100, key="multimodal_voice_prompt")

                if st.button("🎙️ Start Recording"):
                    st.info("🎤 Voice recording feature would be implemented here using speech recognition libraries")

            # Chain of Verification for multimodal
            use_cov_multimodal = st.checkbox("🔍 Apply Chain of Verification to multimodal responses")
            track_multimodal_langfuse = st.checkbox("📊 Track with Langfuse", value=True, key="multimodal_langfuse")

        with col2:
            st.markdown("#### 🚀 Multimodal Results")

            if st.button("🎯 Process Multimodal Input", type="primary"):
                if not multimodal_prompt:
                    st.warning("⚠️ Please provide a prompt.")
                elif not image_files:
                    st.warning("⚠️ Please upload at least one image.")
                else:
                    techniques = PromptingTechniques(
                        st.session_state.model,
                        st.session_state.langfuse if track_multimodal_langfuse else None
                    )

                    for idx, image_file in enumerate(image_files):
                        st.markdown(f"---\n#### 🖼️ Image {idx + 1} Analysis")

                        # Display image
                        col_img, col_result = st.columns([1, 2])

                        with col_img:
                            st.image(image_file, caption=f"Image {idx + 1}", use_column_width=True)

                        with col_result:
                            with st.spinner(f"🤖 Processing Image {idx + 1}..."):
                                try:
                                    # Convert uploaded file to PIL Image
                                    from PIL import Image
                                    image = Image.open(image_file)

                                    # Generate response using Gemini's multimodal capabilities
                                    response = st.session_state.model.generate_content([multimodal_prompt, image])

                                    if use_cov_multimodal:
                                        # Apply Chain of Verification to multimodal response
                                        st.markdown("**🔍 Chain of Verification Applied:**")

                                        # Create a query that includes image context
                                        cov_query = f"Based on the image analysis: {response.text}\n\nOriginal prompt: {multimodal_prompt}"
                                        cov_result = techniques.chain_of_verification(cov_query, "joint", f"multimodal_cov_{idx}")

                                        # Show baseline (original multimodal response)
                                        with st.expander("📝 Original Image Analysis"):
                                            st.write(response.text)

                                        # Show verification process
                                        with st.expander("❓ Verification Questions"):
                                            st.write(cov_result['verification_questions'])

                                        with st.expander("✅ Verification Answers"):
                                            st.write(cov_result['verification_answers'])

                                        # Show final verified response
                                        st.markdown("**🎯 Verified Analysis:**")
                                        st.markdown(f"""
                                        <div class="prompt-result" style="border-left: 4px solid #007bff;">
                                            <p>{cov_result['final_response']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)

                                        final_response = cov_result['final_response']

                                    else:
                                        # Show regular multimodal response
                                        st.markdown(f"""
                                        <div class="prompt-result">
                                            <p>{response.text}</p>
                                        </div>
                                        """, unsafe_allow_html=True)

                                        final_response = response.text

                                    # Save to history
                                    st.session_state.prompts_history.append({
                                        "timestamp": datetime.now(),
                                        "technique": "Multimodal" + (" + CoV" if use_cov_multimodal else ""),
                                        "prompt": f"[Image {idx + 1}] {multimodal_prompt}",
                                        "response": final_response,
                                        "model": selected_model,
                                        "parameters": {
                                            "temperature": temperature,
                                            "max_tokens": max_tokens,
                                            "top_p": top_p,
                                            "multimodal": True,
                                            "chain_of_verification": use_cov_multimodal
                                        },
                                        "langfuse_tracked": track_multimodal_langfuse
                                    })

                                except Exception as e:
                                    st.error(f"❌ Error processing image: {str(e)}")

                    # Display Langfuse link if tracking enabled
                    if track_multimodal_langfuse and st.session_state.langfuse_url:
                        st.markdown(f"""
                        <a href="{st.session_state.langfuse_url}" target="_blank" class="langfuse-link">
                            📊 View Multimodal Processing in Langfuse
                        </a>
                        """, unsafe_allow_html=True)

                    st.success("✅ Multimodal processing completed!")

# Additional features section
st.markdown("---")
st.markdown("## 🎛️ Advanced Features")

# Feature showcase
feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)

with feature_col1:
    st.markdown("""
    <div class="technique-card">
        <h4>⚡ Parallel Execution</h4>
        <p>Execute multiple prompts simultaneously for faster processing and comparative analysis.</p>
    </div>
    """, unsafe_allow_html=True)

with feature_col2:
    st.markdown("""
    <div class="technique-card">
        <h4>📊 Langfuse Integration</h4>
        <p>Comprehensive tracking and observability for all AI interactions with detailed analytics.</p>
    </div>
    """, unsafe_allow_html=True)

with feature_col3:
    st.markdown("""
    <div class="technique-card">
        <h4>🔍 Chain of Verification</h4>
        <p>Improve accuracy through systematic verification of AI responses with self-generated questions.</p>
    </div>
    """, unsafe_allow_html=True)

with feature_col4:
    st.markdown("""
    <div class="technique-card">
        <h4>🧠 Multimodal Intelligence</h4>
        <p>Process images with text prompts and apply verification techniques to visual understanding.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p>🚀 <strong>Advanced Gemini Prompting Studio v2.1</strong> | Built with Streamlit, Google Gemini API & Langfuse</p>
    <p>⚡ Now featuring Parallel Execution for faster processing and comprehensive Langfuse tracking</p>
    <p>🔬 Explore cutting-edge prompting techniques: Zero-shot, Few-shot, CoT, ReAct, CoV, Parallel & Multimodal</p>
</div>
""", unsafe_allow_html=True)