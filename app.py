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
import concurrent.futures
import threading
from langfuse import Langfuse
import uuid

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

    .parallel-result {
        background: #0a0e1a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
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

    .langfuse-link {
        background: #e7f3ff;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #0066cc;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
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
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Sidebar for configuration
with st.sidebar:
    st.markdown("## 🔧 Configuration")

    # User ID
    user_id = st.text_input("👤 User ID", value=st.session_state.user_id, help="Unique identifier for tracing")
    st.session_state.user_id = user_id

    # API Keys
    api_key = st.text_input("🔑 Gemini API Key", type="password", help="Enter your Google Gemini API key")
    
    # Langfuse Configuration
    st.markdown("### 🔍 Langfuse Tracing")
    langfuse_secret_key = st.text_input("🔐 Langfuse Secret Key", type="password")
    langfuse_public_key = st.text_input("🔓 Langfuse Public Key")
    langfuse_host = st.text_input("🌐 Langfuse Host", value="https://cloud.langfuse.com", help="Langfuse server URL")
    
    if langfuse_secret_key and langfuse_public_key:
        try:
            st.session_state.langfuse_client = Langfuse(
                secret_key=langfuse_secret_key,
                public_key=langfuse_public_key,
                host=langfuse_host
            )
            st.success("✅ Langfuse connected!")
            
            # Display Langfuse dashboard link
            st.markdown(f"""
            <div class="langfuse-link">
                🔍 <a href="{langfuse_host}" target="_blank">Open Langfuse Dashboard</a>
                <br>📊 Track User: <code>{user_id}</code>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Langfuse connection failed: {str(e)}")

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
     <h1>🚀 Advanced Gemini Prompting Studio v3.0</h1>
     <p>✨ Now with Parallel Prompting & Langfuse Integration</p>
</div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🎯 Prompting Techniques",
    "🔍 Chain of Verification", 
    "⚡ Parallel Prompting",
    "🔄 Prompt Converter",
    "🧪 Prompt Testing & Evaluation",
    "📊 Analytics Dashboard",
    "📚 Template Library",
    "📥 Export & Import",
    "🧠 Multimodal Prompting"
])

# Enhanced Prompting technique implementations with fixed Langfuse tracing
class PromptingTechniques:
    def __init__(self, model, langfuse_client=None, user_id=None):
        self.model = model
        self.langfuse_client = langfuse_client
        self.user_id = user_id

    def _trace_prompt(self, name: str, prompt: str, technique: str, **kwargs):
        """Create Langfuse trace for prompt execution - Fixed version"""
        if self.langfuse_client:
            try:
                # Create trace using the correct Langfuse v2 API
                trace = self.langfuse_client.trace(
                    name=name,
                    user_id=self.user_id,
                    metadata={
                        "technique": technique,
                        "model": kwargs.get("model", selected_model),
                        "temperature": kwargs.get("temperature", temperature),
                        "max_tokens": kwargs.get("max_tokens", max_tokens),
                        "prompt_length": len(prompt)
                    }
                )
                
                # Create generation within the trace
                generation = trace.generation(
                    name=f"{technique}_generation",
                    model=kwargs.get("model", selected_model),
                    model_parameters={
                        "temperature": kwargs.get("temperature", temperature),
                        "max_tokens": kwargs.get("max_tokens", max_tokens),
                        "top_p": kwargs.get("top_p", top_p)
                    },
                    input=prompt
                )
                
                return trace, generation
            except Exception as e:
                st.warning(f"⚠️ Langfuse tracing warning: {str(e)}")
                return None, None
        return None, None

    def _end_trace(self, trace, generation, response: str, **kwargs):
        """End Langfuse trace with response - Fixed version"""
        if trace and generation and self.langfuse_client:
            try:
                # Update generation with output
                generation.end(output=response)
                
                # Update trace
                trace.update(
                    output=response,
                    metadata={
                        "response_length": len(response),
                        "success": True
                    }
                )
            except Exception as e:
                st.warning(f"⚠️ Langfuse trace update warning: {str(e)}")

    def zero_shot(self, prompt: str) -> str:
        """Zero-shot prompting - direct question without examples"""
        trace, generation = self._trace_prompt("zero_shot_prompting", prompt, "Zero-Shot")
        response = self._generate_response(prompt)
        self._end_trace(trace, generation, response)
        return response

    def few_shot(self, task_description: str, examples: List[Dict], query: str) -> str:
        """Few-shot prompting with examples"""
        prompt = f"{task_description}\n\nExamples:\n"
        for i, example in enumerate(examples, 1):
            prompt += f"{i}. Input: {example['input']} → Output: {example['output']}\n"
        prompt += f"\nNow solve: {query}"
        
        trace, generation = self._trace_prompt("few_shot_prompting", prompt, "Few-Shot")
        response = self._generate_response(prompt)
        self._end_trace(trace, generation, response)
        return response

    def instruction_style(self, instruction: str, content: str) -> str:
        """Direct instruction-based prompting"""
        prompt = f"{instruction}\n\nContent: {content}"
        trace, generation = self._trace_prompt("instruction_prompting", prompt, "Instruction-Style")
        response = self._generate_response(prompt)
        self._end_trace(trace, generation, response)
        return response

    def chain_of_thought(self, problem: str) -> str:
        """Chain-of-thought prompting for step-by-step reasoning"""
        prompt = f"{problem}\n\nLet's think step by step:"
        trace, generation = self._trace_prompt("chain_of_thought", prompt, "Chain-of-Thought")
        response = self._generate_response(prompt)
        self._end_trace(trace, generation, response)
        return response

    def role_based(self, role: str, task: str) -> str:
        """Role-based prompting with specific persona"""
        prompt = f"You are {role}. {task}"
        trace, generation = self._trace_prompt("role_based_prompting", prompt, "Role-Based")
        response = self._generate_response(prompt)
        self._end_trace(trace, generation, response)
        return response

    def react_prompting(self, task: str, tools_available: List[str]) -> str:
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
        trace, generation = self._trace_prompt("react_prompting", prompt, "ReAct")
        response = self._generate_response(prompt)
        self._end_trace(trace, generation, response)
        return response

    def parallel_prompting(self, base_prompt: str, variations: List[str], max_workers: int = 3) -> Dict[str, Any]:
        """Execute multiple prompt variations in parallel"""
        trace = None
        if self.langfuse_client:
            try:
                trace = self.langfuse_client.trace(
                    name="parallel_prompting",
                    user_id=self.user_id,
                    metadata={
                        "technique": "Parallel Prompting",
                        "num_variations": len(variations),
                        "base_prompt": base_prompt
                    }
                )
            except Exception as e:
                st.warning(f"⚠️ Parallel prompting trace warning: {str(e)}")

        def execute_prompt_variation(variation, index):
            """Execute a single prompt variation"""
            full_prompt = f"{base_prompt}\n\n{variation}"
            
            generation = None
            if trace and self.langfuse_client:
                try:
                    generation = trace.generation(
                        name=f"parallel_variation_{index}",
                        model=selected_model,
                        input=full_prompt
                    )
                except Exception as e:
                    st.warning(f"⚠️ Generation trace warning: {str(e)}")
            
            start_time = time.time()
            response = self._generate_response(full_prompt)
            execution_time = time.time() - start_time
            
            if generation and self.langfuse_client:
                try:
                    generation.end(output=response)
                except Exception as e:
                    st.warning(f"⚠️ Generation update warning: {str(e)}")
            
            return {
                "variation": variation,
                "prompt": full_prompt,
                "response": response,
                "execution_time": execution_time,
                "index": index
            }

        # Execute variations in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(execute_prompt_variation, var, i) 
                      for i, var in enumerate(variations)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    st.error(f"Error in parallel execution: {str(e)}")

        # Sort results by index to maintain order
        results.sort(key=lambda x: x['index'])

        # Calculate metrics
        total_time = sum(r['execution_time'] for r in results)
        avg_time = total_time / len(results) if results else 0
        response_lengths = [len(r['response']) for r in results]
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

        parallel_result = {
            "base_prompt": base_prompt,
            "variations": variations,
            "results": results,
            "metrics": {
                "total_execution_time": total_time,
                "average_execution_time": avg_time,
                "average_response_length": avg_length,
                "num_variations": len(variations)
            }
        }

        if trace and self.langfuse_client:
            try:
                trace.update(
                    output=json.dumps(parallel_result, default=str),
                    metadata={
                        "total_execution_time": total_time,
                        "success": True
                    }
                )
            except Exception as e:
                st.warning(f"⚠️ Parallel trace update warning: {str(e)}")

        return parallel_result

    def chain_of_verification(self, query: str, verification_style: str = "joint") -> Dict[str, str]:
        """Chain of Verification prompting for improved accuracy"""
        trace, _ = self._trace_prompt("chain_of_verification", query, "Chain-of-Verification")

        # Step 1: Generate baseline response
        baseline_response = self._generate_response(query)

        # Step 2: Generate verification questions
        verification_prompt = f"""Given the following question and answer, generate specific verification questions that would help check the accuracy and completeness of the answer. Focus on factual claims, logical consistency, and potential gaps.

Original Question: {query}
Answer: {baseline_response}

Please generate 3-5 specific verification questions that would help validate this answer:"""

        verification_questions = self._generate_response(verification_prompt)

        # Step 3: Answer verification questions
        if verification_style == "joint":
            joint_prompt = f"""Please answer the following verification questions based on your knowledge:

{verification_questions}

Provide detailed answers to each question:"""
            verification_answers = self._generate_response(joint_prompt)
        else:
            questions_list = [q.strip() for q in verification_questions.split('\n') if q.strip() and ('?' in q)]
            verification_answers = ""
            for i, question in enumerate(questions_list[:5], 1):
                answer = self._generate_response(f"Answer this question concisely: {question}")
                verification_answers += f"Q{i}: {question}\nA{i}: {answer}\n\n"

        # Step 4: Generate final verified response
        final_prompt = f"""Based on the original question, initial answer, verification questions, and their answers, provide a final, improved, and verified response. Correct any errors found and incorporate insights from the verification process.

Original Question: {query}
Initial Answer: {baseline_response}

Verification Questions: {verification_questions}
Verification Answers: {verification_answers}

Final Verified Response:"""

        final_response = self._generate_response(final_prompt)

        result = {
            "baseline_response": baseline_response,
            "verification_questions": verification_questions,
            "verification_answers": verification_answers,
            "final_response": final_response,
            "verification_style": verification_style
        }

        self._end_trace(trace, None, json.dumps(result, default=str))
        return result

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
                conversion_prompts[conversion_key] = f"""Convert the following {source_technique} prompt into a {target_technique} prompt:

Original prompt: {original_prompt}

Adapt the prompt structure and format to match the {target_technique} technique while preserving the core intent."""

        prompt_to_convert = conversion_prompts.get(conversion_key, conversion_prompts["any_to_cov"])
        
        trace, generation = self._trace_prompt("prompt_conversion", prompt_to_convert, "Prompt Conversion")
        response = self._generate_response(prompt_to_convert)
        self._end_trace(trace, generation, response)
        
        return response

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

        analysis_result = self._generate_response(analysis_prompt)

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

        detected_technique = self._generate_response(technique_detection_prompt)

        return {
            "analysis": analysis_result,
            "detected_technique": detected_technique.strip(),
            "prompt": prompt
        }

    def _generate_response(self, prompt: str) -> str:
        """Helper method to generate response from model"""
        try:
            if not self.model:
                return "❌ Model not initialized. Please provide API key."

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

# Template library (updated with parallel prompting templates)
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
        "Math Problem": "A train travels at {speed} km/h for {time} hours. How far does it travel? Let's think step by step:",
        "Logic Puzzle": "If all cats are animals, and some animals are pets, can we conclude that some cats are pets? Let's analyze this step by step:",
        "Word Problem": "Sarah has {initial_amount} apples. She gives {given_amount} to her friend. How many does she have left? Let's solve this step by step:"
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
    "Parallel Prompting": {
        "Creative Variations": {
            "base": "Write a short story about {topic}",
            "variations": [
                "Focus on character development and emotional depth",
                "Emphasize action and fast-paced narrative",
                "Use a mysterious and suspenseful tone",
                "Write from multiple perspectives"
            ]
        },
        "Problem Solving": {
            "base": "Solve this problem: {problem}",
            "variations": [
                "Use a mathematical approach",
                "Apply creative thinking and analogies",
                "Consider practical constraints and limitations",
                "Think about long-term implications"
            ]
        },
        "Analysis Approaches": {
            "base": "Analyze {subject}",
            "variations": [
                "From a technical perspective",
                "From a business impact viewpoint", 
                "Considering social and ethical implications",
                "From a user experience standpoint"
            ]
        }
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

    with col2:
        st.markdown("### 🚀 Results")

        if st.button("🎯 Generate Response", type="primary"):
            if not st.session_state.model:
                st.error("❌ Please configure your API key first!")
            else:
                with st.spinner("🤖 Generating response..."):
                    techniques = PromptingTechniques(
                        st.session_state.model, 
                        st.session_state.langfuse_client,
                        st.session_state.user_id
                    )

                    # Generate response based on technique
                    if technique == "Zero-Shot" and prompt_input:
                        response = techniques.zero_shot(prompt_input)
                        used_prompt = prompt_input

                    elif technique == "Few-Shot" and task_desc and examples and query:
                        response = techniques.few_shot(task_desc, examples, query)
                        used_prompt = f"Task: {task_desc}, Examples: {examples}, Query: {query}"

                    elif technique == "Instruction-Style" and instruction and content:
                        response = techniques.instruction_style(instruction, content)
                        used_prompt = f"Instruction: {instruction}, Content: {content}"

                    elif technique == "Chain-of-Thought" and prompt_input:
                        response = techniques.chain_of_thought(prompt_input)
                        used_prompt = prompt_input

                    elif technique == "Role-Based" and role and task:
                        response = techniques.role_based(role, task)
                        used_prompt = f"Role: {role}, Task: {task}"

                    elif technique == "ReAct" and task_input and tools:
                        response = techniques.react_prompting(task_input, tools)
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

                        # Display Langfuse trace link if available
                        if st.session_state.langfuse_client:
                            st.markdown(f"""
                            <div class="langfuse-link">
                                🔍 <strong>Langfuse Trace:</strong> Check your <a href="{langfuse_host}" target="_blank">Langfuse Dashboard</a> for detailed trace information
                                <br>👤 User ID: <code>{st.session_state.user_id}</code>
                            </div>
                            """, unsafe_allow_html=True)

                        # Save to history
                        st.session_state.prompts_history.append({
                            "timestamp": datetime.now(),
                            "technique": technique,
                            "prompt": used_prompt,
                            "response": response,
                            "model": selected_model,
                            "user_id": st.session_state.user_id,
                            "parameters": {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
                        })

                        st.success("✅ Response saved to history!")

# Tab 2: Chain of Verification
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
                        st.session_state.langfuse_client,
                        st.session_state.user_id
                    )

                    # Run CoV process
                    cov_result = techniques.chain_of_verification(cov_query, verification_style)

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

                    # Display Langfuse trace link if available
                    if st.session_state.langfuse_client:
                        st.markdown(f"""
                        <div class="langfuse-link">
                            🔍 <strong>Langfuse Trace:</strong> Check your <a href="{langfuse_host}" target="_blank">Langfuse Dashboard</a> for detailed CoV trace
                            <br>👤 User ID: <code>{st.session_state.user_id}</code>
                        </div>
                        """, unsafe_allow_html=True)

                    # Save to history
                    st.session_state.prompts_history.append({
                        "timestamp": datetime.now(),
                        "technique": "Chain-of-Verification",
                        "prompt": cov_query,
                        "response": cov_result['final_response'],
                        "model": selected_model,
                        "user_id": st.session_state.user_id,
                        "parameters": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "verification_style": verification_style,
                            "baseline_response": cov_result['baseline_response'],
                            "verification_questions": cov_result['verification_questions'],
                            "verification_answers": cov_result['verification_answers']
                        }
                    })

                    st.success("✅ Chain of Verification completed and saved to history!")

# Tab 3: Parallel Prompting - NEW TAB
with tab3:
    st.markdown("### ⚡ Parallel Prompting")

    st.info("""
    **Parallel Prompting** allows you to:
    🚀 Execute multiple prompt variations simultaneously
    📊 Compare responses from different approaches
    ⏱️ Reduce total execution time through parallelization
    🔍 Analyze variation effectiveness and consistency
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 🎯 Configuration")

        # Base prompt input
        base_prompt = st.text_area("📝 Base Prompt:",
                                 placeholder="Enter the core prompt that will be enhanced with variations...",
                                 height=100)

        # Template selection
        st.markdown("#### 📚 Use Template")
        parallel_template = st.selectbox(
            "Select parallel template:",
            ["None"] + list(TEMPLATE_LIBRARY["Parallel Prompting"].keys())
        )

        variations = []
        
        if parallel_template != "None":
            template = TEMPLATE_LIBRARY["Parallel Prompting"][parallel_template]
            
            # Fill template variables if needed
            template_base = template["base"]
            variables = re.findall(r'\{(\w+)\}', template_base)
            
            if variables:
                st.markdown("**Fill template variables:**")
                var_values = {}
                for var in variables:
                    var_values[var] = st.text_input(f"{var.replace('_', ' ').title()}:", key=f"parallel_var_{var}")
                
                if all(var_values.values()):
                    base_prompt = template_base.format(**var_values)
                    variations = template["variations"]
                    
                    st.text_area("Generated Base Prompt:", value=base_prompt, height=80, disabled=True)
                    
                    st.markdown("**Template Variations:**")
                    for i, var in enumerate(variations, 1):
                        st.text(f"{i}. {var}")
        
        # Manual variations input
        st.markdown("#### ✋ Manual Variations")
        num_variations = st.number_input("Number of variations:", 1, 8, min(4, len(variations)) if variations else 4)
        
        # If template variations exist, use them as defaults
        manual_variations = []
        for i in range(num_variations):
            default_val = variations[i] if i < len(variations) else ""
            var_text = st.text_area(f"Variation {i+1}:", 
                                   value=default_val,
                                   height=60, 
                                   key=f"manual_var_{i}")
            if var_text.strip():
                manual_variations.append(var_text.strip())

        # Use manual variations if provided, otherwise use template
        final_variations = manual_variations if manual_variations else variations

        # Execution settings
        st.markdown("#### ⚙️ Execution Settings")
        max_workers = st.slider("🔄 Max Parallel Workers:", 1, 5, 3)
        
        # Analysis options
        st.markdown("#### 📊 Analysis Options")
        analyze_responses = st.checkbox("📈 Analyze response patterns", value=True)
        compare_lengths = st.checkbox("📏 Compare response lengths", value=True)
        show_timing = st.checkbox("⏱️ Show execution timing", value=True)

    with col2:
        st.markdown("#### ⚡ Parallel Execution Results")

        if st.button("🚀 Execute Parallel Prompts", type="primary"):
            if not base_prompt:
                st.warning("⚠️ Please provide a base prompt!")
            elif not final_variations:
                st.warning("⚠️ Please provide at least one variation!")
            elif not st.session_state.model:
                st.error("❌ Please configure your API key first!")
            else:
                with st.spinner(f"⚡ Executing {len(final_variations)} prompt variations in parallel..."):
                    techniques = PromptingTechniques(
                        st.session_state.model,
                        st.session_state.langfuse_client,
                        st.session_state.user_id
                    )

                    # Execute parallel prompting
                    start_time = time.time()
                    parallel_result = techniques.parallel_prompting(base_prompt, final_variations, max_workers)
                    total_time = time.time() - start_time

                    # Display results
                    st.markdown("### 📊 Execution Summary")
                    
                    # Metrics
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
                    with metrics_col1:
                        st.metric("🔢 Variations", len(final_variations))
                    
                    with metrics_col2:
                        st.metric("⏱️ Total Time", f"{total_time:.2f}s")
                    
                    with metrics_col3:
                        avg_time = parallel_result["metrics"]["average_execution_time"]
                        st.metric("⏳ Avg Time", f"{avg_time:.2f}s")
                    
                    with metrics_col4:
                        avg_length = parallel_result["metrics"]["average_response_length"]
                        st.metric("📝 Avg Length", f"{avg_length:.0f}")

                    # Display Langfuse trace link if available
                    if st.session_state.langfuse_client:
                        st.markdown(f"""
                        <div class="langfuse-link">
                            🔍 <strong>Langfuse Trace:</strong> Check your <a href="{langfuse_host}" target="_blank">Langfuse Dashboard</a> for detailed parallel execution trace
                            <br>👤 User ID: <code>{st.session_state.user_id}</code>
                        </div>
                        """, unsafe_allow_html=True)

                    # Individual results
                    st.markdown("### 📋 Individual Results")
                    
                    for i, result in enumerate(parallel_result["results"], 1):
                        with st.expander(f"🔸 Variation {i}: {result['variation'][:50]}..."):
                            st.markdown(f"**Variation:** {result['variation']}")
                            st.markdown(f"**Execution Time:** {result['execution_time']:.2f}s")
                            st.markdown("**Response:**")
                            st.markdown(f"""
                            <div class="parallel-result">
                                <p>{result['response']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Analysis section
                    if analyze_responses or compare_lengths or show_timing:
                        st.markdown("### 📈 Analysis")
                        
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            if show_timing:
                                # Execution timing chart
                                timing_data = pd.DataFrame([
                                    {"Variation": f"Var {i+1}", "Time": result["execution_time"]}
                                    for i, result in enumerate(parallel_result["results"])
                                ])
                                fig_timing = px.bar(timing_data, x="Variation", y="Time", 
                                                  title="Execution Time by Variation")
                                st.plotly_chart(fig_timing, use_container_width=True)
                        
                        with analysis_col2:
                            if compare_lengths:
                                # Response length comparison
                                length_data = pd.DataFrame([
                                    {"Variation": f"Var {i+1}", "Length": len(result["response"])}
                                    for i, result in enumerate(parallel_result["results"])
                                ])
                                fig_length = px.bar(length_data, x="Variation", y="Length",
                                                  title="Response Length by Variation")
                                st.plotly_chart(fig_length, use_container_width=True)

                        if analyze_responses:
                            st.markdown("#### 🔍 Response Pattern Analysis")
                            
                            # Word frequency analysis
                            all_responses = " ".join([result["response"] for result in parallel_result["results"]])
                            words = all_responses.lower().split()
                            word_freq = pd.Series(words).value_counts().head(10)
                            
                            fig_words = px.bar(x=word_freq.values, y=word_freq.index,
                                             orientation='h', title="Top 10 Most Common Words")
                            st.plotly_chart(fig_words, use_container_width=True)

                            # Response similarity analysis
                            st.markdown("#### 🔗 Response Similarity")
                            similarity_data = []
                            for i, result1 in enumerate(parallel_result["results"]):
                                for j, result2 in enumerate(parallel_result["results"]):
                                    if i < j:  # Avoid duplicate pairs
                                        # Simple word overlap similarity
                                        words1 = set(result1["response"].lower().split())
                                        words2 = set(result2["response"].lower().split())
                                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                                        similarity_data.append({
                                            "Pair": f"Var{i+1} vs Var{j+1}",
                                            "Similarity": similarity
                                        })
                            
                            if similarity_data:
                                similarity_df = pd.DataFrame(similarity_data)
                                fig_sim = px.bar(similarity_df, x="Pair", y="Similarity",
                                               title="Response Similarity Between Variations")
                                st.plotly_chart(fig_sim, use_container_width=True)

                    # Best response selection
                    st.markdown("### 🏆 Response Selection")
                    
                    selection_method = st.radio(
                        "Select best response by:",
                        ["Manual Review", "Longest Response", "Fastest Generation", "Most Detailed"]
                    )
                    
                    if selection_method == "Longest Response":
                        best_result = max(parallel_result["results"], key=lambda x: len(x["response"]))
                        st.info(f"🏆 Longest response: Variation {best_result['index'] + 1}")
                    elif selection_method == "Fastest Generation":
                        best_result = min(parallel_result["results"], key=lambda x: x["execution_time"])
                        st.info(f"🏆 Fastest response: Variation {best_result['index'] + 1}")
                    elif selection_method == "Most Detailed":
                        # Simple heuristic: most sentences
                        best_result = max(parallel_result["results"], 
                                        key=lambda x: len([s for s in x["response"].split('.') if s.strip()]))
                        st.info(f"🏆 Most detailed response: Variation {best_result['index'] + 1}")
                    else:
                        best_result = None
                        st.info("👆 Review responses above and manually select the best one")

                    # Save to history
                    st.session_state.prompts_history.append({
                        "timestamp": datetime.now(),
                        "technique": "Parallel Prompting",
                        "prompt": f"Base: {base_prompt} | Variations: {len(final_variations)}",
                        "response": json.dumps(parallel_result, default=str),
                        "model": selected_model,
                        "user_id": st.session_state.user_id,
                        "parameters": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "max_workers": max_workers,
                            "num_variations": len(final_variations),
                            "total_execution_time": total_time,
                            "parallel_metrics": parallel_result["metrics"]
                        }
                    })

                    st.success("✅ Parallel prompting completed and saved to history!")

# Tab 4: Prompt Converter
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
                            st.session_state.langfuse_client,
                            st.session_state.user_id
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

                        # Display Langfuse trace link if available
                        if st.session_state.langfuse_client:
                            st.markdown(f"""
                            <div class="langfuse-link">
                                🔍 <strong>Langfuse Trace:</strong> Check your <a href="{langfuse_host}" target="_blank">Langfuse Dashboard</a> for prompt conversion trace
                                <br>👤 User ID: <code>{st.session_state.user_id}</code>
                            </div>
                            """, unsafe_allow_html=True)

                        # Test both prompts option
                        if st.button("⚡ Test Both Prompts"):
                            st.markdown("##### 🧪 Comparison Results")

                            col_orig, col_conv = st.columns(2)

                            with col_orig:
                                st.markdown("**Original Response:**")
                                orig_response = techniques.zero_shot(input_prompt)
                                st.write(orig_response)

                            with col_conv:
                                st.markdown("**Converted Response:**")
                                conv_response = techniques.zero_shot(converted_prompt)
                                st.write(conv_response)

                        # Save to history
                        st.session_state.prompts_history.append({
                            "timestamp": datetime.now(),
                            "technique": f"Conversion: {source_tech} → {target_technique}",
                            "prompt": converted_prompt,
                            "response": f"Converted from: {input_prompt}",
                            "model": selected_model,
                            "user_id": st.session_state.user_id,
                            "parameters": {
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "top_p": top_p,
                                "conversion_context": context,
                                "source_technique": source_tech,
                                "target_technique": target_technique
                            }
                        })

                        st.success("✅ Prompt converted and saved to history!")

    with converter_tab2:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### 📊 Prompt Analysis")

            # Input prompt for analysis
            analysis_prompt = st.text_area("📝 Enter prompt to analyze:",
                                         placeholder="Enter any prompt for detailed analysis...",
                                         height=120)

            # Analysis options
            analysis_depth = st.radio("🔍 Analysis Depth:",
                                    ["Quick", "Detailed", "Comprehensive"])

            focus_areas = st.multiselect("🎯 Focus Areas:",
                                       ["Clarity", "Specificity", "Structure", "Effectiveness",
                                        "Potential Issues", "Improvement Suggestions"],
                                       default=["Clarity", "Effectiveness"])

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
                            st.session_state.langfuse_client,
                            st.session_state.user_id
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
                            "user_id": st.session_state.user_id,
                            "parameters": {
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "top_p": top_p,
                                "detected_technique": analysis_result['detected_technique'],
                                "analysis_depth": analysis_depth,
                                "focus_areas": focus_areas
                            }
                        })

                        st.success("✅ Analysis completed and saved to history!")

# Tab 5: Prompt Testing & Evaluation
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

            if st.button("🧪 Run Batch Tests"):
                if not st.session_state.model:
                    st.error("❌ Please configure your API key first!")
                else:
                    results = []
                    progress_bar = st.progress(0)

                    techniques = PromptingTechniques(
                        st.session_state.model,
                        st.session_state.langfuse_client,
                        st.session_state.user_id
                    )

                    for idx, row in df.iterrows():
                        prompt = row.get('prompt', '')
                        expected = row.get('expected', '')

                        if prompt:
                            response = techniques.zero_shot(prompt)

                            results.append({
                                'prompt': prompt,
                                'expected': expected,
                                'actual': response,
                                'timestamp': datetime.now()
                            })

                        progress_bar.progress((idx + 1) / len(df))

                    st.session_state.evaluation_results = results
                    st.success(f"✅ Completed {len(results)} tests!")
                    
                    # Display Langfuse trace link
                    if st.session_state.langfuse_client:
                        st.markdown(f"""
                        <div class="langfuse-link">
                            🔍 <strong>Batch Testing Traces:</strong> Check your <a href="{langfuse_host}" target="_blank">Langfuse Dashboard</a> for all test traces
                            <br>👤 User ID: <code>{st.session_state.user_id}</code>
                        </div>
                        """, unsafe_allow_html=True)

        # Manual testing
        st.markdown("#### ✋ Manual Testing")
        test_prompt = st.text_area("🧪 Test Prompt:", height=100)
        expected_output = st.text_area("🎯 Expected Output:", height=100)

        if st.button("🧪 Run Single Test"):
            if test_prompt and st.session_state.model:
                techniques = PromptingTechniques(
                    st.session_state.model,
                    st.session_state.langfuse_client,
                    st.session_state.user_id
                )
                actual_output = techniques.zero_shot(test_prompt)

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
                
                # Display Langfuse trace link
                if st.session_state.langfuse_client:
                    st.markdown(f"""
                    <div class="langfuse-link">
                        🔍 <strong>Single Test Trace:</strong> Check your <a href="{langfuse_host}" target="_blank">Langfuse Dashboard</a>
                        <br>👤 User ID: <code>{st.session_state.user_id}</code>
                    </div>
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

# Tab 6: Analytics Dashboard
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
            models_used = df_history['model'].nunique()
            st.metric("🤖 Models Used", models_used)

        # Display current user info
        st.markdown(f"""
        <div class="langfuse-link">
            👤 <strong>Current User:</strong> <code>{st.session_state.user_id}</code>
            <br>🔍 <strong>Langfuse Dashboard:</strong> <a href="{langfuse_host}" target="_blank">View All Traces</a>
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

        # Response length by technique
        fig_box = px.box(df_history, x='technique', y=df_history['response'].str.len(),
                        title="Response Length by Technique")
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

        # Recent prompts table
        st.markdown("#### 📝 Recent Prompts")
        recent_prompts = df_history.tail(10)[['timestamp', 'technique', 'prompt', 'model', 'user_id']].copy()
        recent_prompts['prompt'] = recent_prompts['prompt'].str[:100] + '...'
        st.dataframe(recent_prompts, use_container_width=True)

    else:
        st.info("📊 No data available yet. Start using prompting techniques to see analytics!")

# Tab 7: Template Library
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
                            var_values[var] = st.text_input(f"{var}:", key=f"var_{var}")

                        if st.button("📝 Generate Prompt"):
                            filled_template = template.format(**var_values)
                            st.session_state.filled_template = filled_template

                elif isinstance(template, dict):
                    # Complex template with examples or variations
                    if 'base' in template and 'variations' in template:
                        # Parallel prompting template
                        st.markdown("**Base Template:**")
                        st.code(template['base'], language="text")
                        
                        st.markdown("**Variations:**")
                        for i, var in enumerate(template['variations'], 1):
                            st.write(f"{i}. {var}")
                    else:
                        # Few-shot template
                        st.markdown("**Description:**")
                        st.write(template.get('description', ''))

                        if 'examples' in template:
                            st.markdown("**Examples:**")
                            for i, example in enumerate(template['examples'], 1):
                                st.write(f"{i}. Input: {example['input']} → Output: {example['output']}")

        with col2:
            st.markdown("### 🎯 Generated Prompt")

            if hasattr(st.session_state, 'filled_template'):
                st.markdown("**Ready to use:**")
                st.code(st.session_state.filled_template, language="text")

                if st.button("🚀 Execute Prompt"):
                    if st.session_state.model:
                        techniques = PromptingTechniques(
                            st.session_state.model,
                            st.session_state.langfuse_client,
                            st.session_state.user_id
                        )
                        response = techniques.zero_shot(st.session_state.filled_template)

                        st.markdown("**Response:**")
                        st.markdown(f"""
                        <div class="prompt-result">
                            <p>{response}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display Langfuse trace link
                        if st.session_state.langfuse_client:
                            st.markdown(f"""
                            <div class="langfuse-link">
                                🔍 <strong>Template Execution Trace:</strong> <a href="{langfuse_host}" target="_blank">View in Langfuse</a>
                                <br>👤 User ID: <code>{st.session_state.user_id}</code>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("👆 Select a template and fill variables to generate prompt")

# Tab 8: Export & Import
with tab8:
    st.markdown("### 📥 Export & Import")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📤 Export Data")

        export_type = st.selectbox("📁 Export Format:", ["JSON", "CSV", "TXT"])

        if st.session_state.prompts_history:
            if st.button("📥 Download History"):
                df_export = pd.DataFrame(st.session_state.prompts_history)

                if export_type == "CSV":
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"prompts_history_{st.session_state.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                elif export_type == "JSON":
                    json_data = df_export.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"prompts_history_{st.session_state.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                elif export_type == "TXT":
                    txt_content = f"Prompts History for User: {st.session_state.user_id}\n"
                    txt_content += f"Export Date: {datetime.now()}\n"
                    txt_content += "=" * 60 + "\n\n"
                    
                    for item in st.session_state.prompts_history:
                        txt_content += f"Timestamp: {item['timestamp']}\n"
                        txt_content += f"Technique: {item['technique']}\n"
                        txt_content += f"Prompt: {item['prompt']}\n"
                        txt_content += f"Response: {item['response']}\n"
                        txt_content += f"User ID: {item.get('user_id', 'N/A')}\n"
                        txt_content += "-" * 50 + "\n\n"

                    st.download_button(
                        label="📥 Download TXT",
                        data=txt_content,
                        file_name=f"prompts_history_{st.session_state.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("📊 No history data to export!")

        # Export Langfuse data info
        if st.session_state.langfuse_client:
            st.markdown("#### 🔍 Langfuse Data")
            st.info(f"""
            **Langfuse Integration Active**
            
            - User ID: `{st.session_state.user_id}`
            - All prompts are automatically traced
            - View detailed analytics in [Langfuse Dashboard]({langfuse_host})
            - Export traces directly from Langfuse for advanced analysis
            """)

    with col2:
        st.markdown("#### 📤 Import Data")

        uploaded_history = st.file_uploader("📁 Upload History File", type=['json', 'csv'])

        if uploaded_history:
            try:
                if uploaded_history.name.endswith('.json'):
                    imported_data = json.load(uploaded_history)
                    # Add user_id to imported data if missing
                    for item in imported_data:
                        if 'user_id' not in item:
                            item['user_id'] = st.session_state.user_id
                    st.session_state.prompts_history.extend(imported_data)
                    st.success(f"✅ Imported {len(imported_data)} records!")

                elif uploaded_history.name.endswith('.csv'):
                    df_imported = pd.read_csv(uploaded_history)
                    imported_data = df_imported.to_dict('records')
                    # Add user_id to imported data if missing
                    for item in imported_data:
                        if 'user_id' not in item:
                            item['user_id'] = st.session_state.user_id
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
            st.info("💡 Note: Langfuse traces are preserved in the cloud and can be accessed via the dashboard")

# Tab 9: Multimodal Prompting
with tab9:
    st.markdown("### 🧠 Multimodal Prompting")

    if not st.session_state.model:
        st.warning("⚠️ Please configure your Gemini API key in the sidebar.")
    else:
        st.info("📸 Upload images and provide text prompts for multimodal AI interactions")

        col1, col2 = st.columns([1, 2])

        with col1:
            # File uploads
            image_files = st.file_uploader("📸 Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

            # Prompt input
            multimodal_prompt = st.text_area("💭 Enter your prompt:", height=100, key="multimodal_text_prompt")

            # Chain of Verification for multimodal
            use_cov_multimodal = st.checkbox("🔍 Apply Chain of Verification to multimodal responses")

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
                        st.session_state.langfuse_client,
                        st.session_state.user_id
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

                                    # Create Langfuse trace for multimodal
                                    if st.session_state.langfuse_client:
                                        try:
                                            trace = st.session_state.langfuse_client.trace(
                                                name=f"multimodal_analysis_image_{idx + 1}",
                                                user_id=st.session_state.user_id,
                                                metadata={
                                                    "technique": "Multimodal" + (" + CoV" if use_cov_multimodal else ""),
                                                    "image_index": idx + 1,
                                                    "total_images": len(image_files),
                                                    "prompt": multimodal_prompt
                                                }
                                            )
                                        except Exception as e:
                                            st.warning(f"⚠️ Multimodal trace warning: {str(e)}")
                                            trace = None

                                    # Generate response using Gemini's multimodal capabilities
                                    response = st.session_state.model.generate_content([multimodal_prompt, image])

                                    if use_cov_multimodal:
                                        # Apply Chain of Verification to multimodal response
                                        st.markdown("**🔍 Chain of Verification Applied:**")

                                        # Create a query that includes image context
                                        cov_query = f"Based on the image analysis: {response.text}\n\nOriginal prompt: {multimodal_prompt}"
                                        cov_result = techniques.chain_of_verification(cov_query, "joint")

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

                                    # Update Langfuse trace
                                    if trace and st.session_state.langfuse_client:
                                        try:
                                            trace.update(
                                                output=final_response,
                                                metadata={
                                                    "response_length": len(final_response),
                                                    "success": True
                                                }
                                            )
                                        except Exception as e:
                                            st.warning(f"⚠️ Multimodal trace update warning: {str(e)}")

                                    # Save to history
                                    st.session_state.prompts_history.append({
                                        "timestamp": datetime.now(),
                                        "technique": "Multimodal" + (" + CoV" if use_cov_multimodal else ""),
                                        "prompt": f"[Image {idx + 1}] {multimodal_prompt}",
                                        "response": final_response,
                                        "model": selected_model,
                                        "user_id": st.session_state.user_id,
                                        "parameters": {
                                            "temperature": temperature,
                                            "max_tokens": max_tokens,
                                            "top_p": top_p,
                                            "multimodal": True,
                                            "chain_of_verification": use_cov_multimodal
                                        }
                                    })

                                except Exception as e:
                                    st.error(f"❌ Error processing image: {str(e)}")
                                    
                                    # Log error to Langfuse
                                    if st.session_state.langfuse_client:
                                        try:
                                            error_trace = st.session_state.langfuse_client.trace(
                                                name=f"multimodal_error_image_{idx + 1}",
                                                user_id=st.session_state.user_id,
                                                metadata={
                                                    "error": str(e),
                                                    "technique": "Multimodal Error",
                                                    "image_index": idx + 1
                                                }
                                            )
                                            error_trace.update(output=f"Error: {str(e)}")
                                        except Exception as trace_error:
                                            st.warning(f"⚠️ Error logging warning: {str(trace_error)}")

                    # Display overall Langfuse trace link
                    if st.session_state.langfuse_client:
                        st.markdown(f"""
                        <div class="langfuse-link">
                            🔍 <strong>Multimodal Processing Traces:</strong> Check your <a href="{langfuse_host}" target="_blank">Langfuse Dashboard</a> for detailed image analysis traces
                            <br>👤 User ID: <code>{st.session_state.user_id}</code>
                        </div>
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
        <h4>⚡ Parallel Prompting</h4>
        <p>Execute multiple prompt variations simultaneously to explore different approaches and compare results efficiently.</p>
    </div>
    """, unsafe_allow_html=True)

with feature_col2:
    st.markdown("""
    <div class="technique-card">
        <h4>🔍 Chain of Verification</h4>
        <p>Improve accuracy through systematic verification of AI responses with self-generated questions and answers.</p>
    </div>
    """, unsafe_allow_html=True)

with feature_col3:
    st.markdown("""
    <div class="technique-card">
        <h4>🔍 Langfuse Integration</h4>
        <p>Comprehensive prompt tracing and analytics with user tracking for advanced observability and optimization.</p>
    </div>
    """, unsafe_allow_html=True)

with feature_col4:
    st.markdown("""
    <div class="technique-card">
        <h4>🧠 Multimodal Intelligence</h4>
        <p>Process images with text prompts and apply verification techniques to visual understanding tasks.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p>🤖 <strong>Advanced Gemini Prompting Studio v3.0</strong> | Built with Streamlit & Google Gemini API</p>
    <p>⚡ Now featuring Parallel Prompting for efficient multi-variation execution</p>
    <p>🔍 Integrated with Langfuse for comprehensive prompt tracing and analytics</p>
    <p>👤 Current User: <code>{st.session_state.user_id}</code> | 
       🔗 <a href="{langfuse_host if st.session_state.langfuse_client else '#'}" target="_blank">Langfuse Dashboard</a></p>
    <p>🔬 Explore cutting-edge prompting techniques: Zero-shot, Few-shot, CoT, ReAct, CoV, Parallel & Multimodal</p>
</div>
""", unsafe_allow_html=True)