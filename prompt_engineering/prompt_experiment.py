#!/usr/bin/env python3
"""
Prompt Engineering Experiment - Compare zero-shot, one-shot, few-shot prompt effects
Using GLM4.7 model, test different prompt modes for four agents
"""

import json
import os
import sys
import re
import argparse
from typing import Any, Dict, List, Optional, TypedDict, Literal
from pathlib import Path
from datetime import datetime

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_llm
from src.tools import run_sqlite_query, read_json_file, get_db_schema
from src.eval_utils import compare_query_results, truncate_for_json
from prompt_engineering.prompts_templates import get_prompt_template
from langchain_core.messages import SystemMessage, HumanMessage


class AgentState(TypedDict):
    """Agent state definition"""
    question: str
    db_path: str
    recovery_json_path: str
    initial_query_result: str
    initial_query_sql: str
    has_recovery_json: bool
    workflow_type: Literal["CASE_1_MISSING_TABLE", "CASE_2_NORMAL", "CASE_3_CORRUPTED_DATA", "CASE_4_MISSING_COLUMN"]
    final_answer: str
    final_sql: str
    repair_sql: str


class PromptExperiment:
    """Prompt Engineering Experiment class"""
    
    def __init__(self, prompt_mode: str, output_dir: str = "results/prompt_eng"):
        """
        Initialize prompt engineering experiment
        
        Args:
            prompt_mode: "zero_shot", "one_shot", or "few_shot"
            output_dir: Results output directory
        """
        self.prompt_mode = prompt_mode
        self.output_dir = Path(output_dir) / f"{prompt_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.jsonl"
        self.meta_file = self.output_dir / "meta.json"
        
        # Save experiment metadata
        self._save_metadata()
        
        print(f"Prompt engineering experiment initialized")
        print(f"Prompt mode: {prompt_mode}")
        print(f"Results will be saved to: {self.output_dir}")
    
    def _save_metadata(self):
        """Save experiment metadata"""
        meta = {
            "experiment_type": "prompt_engineering",
            "prompt_mode": self.prompt_mode,
            "model": "glm4.7",
            "start_time": datetime.now().isoformat(),
            "description": f"Prompt engineering experiment - {self.prompt_mode}mode"
        }
        
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    
    def save_result(self, result: Dict[str, Any]):
        """Save single question result"""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    def _extract_sql(self, content: str) -> str:
        """Helper to extract SQL from LLM response"""
        match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return content.replace("```sql", "").replace("```", "").strip()
    
    def _extract_sql_candidates(self, content: str) -> List[str]:
        """Extract all SQL candidates from content"""
        blocks = re.findall(r"```sql\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
        blocks = [b.strip() for b in blocks if b.strip()]
        if blocks:
            return blocks

        upper = content.upper()
        keywords = ["SELECT ", "WITH ", "PRAGMA ", "CREATE ", "INSERT ", "UPDATE ", "ALTER "]
        first_pos = None
        for k in keywords:
            pos = upper.find(k)
            if pos != -1:
                first_pos = pos if first_pos is None else min(first_pos, pos)

        if first_pos is None:
            return []

        candidate = content[first_pos:].strip()
        candidate = candidate.replace("```", "").strip()
        return [candidate] if candidate else []
    
    def _split_sql_statements(self, sql: str) -> List[str]:
        """Split SQL into individual statements"""
        statements: List[str] = []
        buf: List[str] = []
        quote: Optional[str] = None
        i = 0
        while i < len(sql):
            ch = sql[i]
            if quote is not None:
                buf.append(ch)
                if ch == quote:
                    if quote == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                        buf.append(sql[i + 1])
                        i += 2
                        continue
                    quote = None
                i += 1
                continue

            if ch in ("'", '"', "`"):
                quote = ch
                buf.append(ch)
                i += 1
                continue

            if ch == ";":
                stmt = "".join(buf).strip()
                if stmt:
                    statements.append(stmt)
                buf = []
                i += 1
                continue

            buf.append(ch)
            i += 1

        tail = "".join(buf).strip()
        if tail:
            statements.append(tail)
        return statements
    
    def router_node(self, state: AgentState) -> Dict[str, Any]:
        """Router node - Intent recognition"""
        llm = get_llm("glm4.7")
        question = state['question']
        db_path = state['db_path']
        
        # 1. Try executing a query
        gen_sql_prompt = f"Generate a SQLite query for the following question on schema {get_db_schema(db_path)}:\nQuestion: {question}\nOutput ONLY the SQL."
        gen_sql_response = llm.invoke(gen_sql_prompt)
        sql = self._extract_sql(gen_sql_response.content)
        
        print(f"[Router] Initial Try SQL:\n{sql}\n")
        result = run_sqlite_query(db_path, sql)
        
        # 2. Check if recovery JSON exists
        has_json = False
        if state.get('recovery_json_path') and os.path.exists(state['recovery_json_path']):
            try:
                meta = read_json_file(state['recovery_json_path'])
                has_json = bool(meta.get("case_type"))
            except Exception:
                has_json = False
        
        # 3. Prepare Metadata Hint
        recovery_metadata = "Unavailable"
        if has_json:
            try:
                with open(state['recovery_json_path'], 'r') as f:
                    meta = json.load(f)
                    recovery_metadata = json.dumps({k: v for k, v in meta.items() if k != 'data_payload'})
            except:
                pass

        # 4. LLM diagnosis - using specified Prompt mode
        result_str = str(result)
        if len(result_str) > 2000:
            result_str = result_str[:2000] + "... [TRUNCATED]"
        
        router_prompt = get_prompt_template(self.prompt_mode, "router")
        diagnosis_msg = router_prompt.format(
            question=question,
            result=result_str,
            has_recovery_json=has_json,
            recovery_metadata=recovery_metadata
        )
        
        decision = llm.invoke([HumanMessage(content=diagnosis_msg)]).content.strip()
        
        # Clean LLM output
        workflow_type = "CASE_2_NORMAL"
        if "CASE_1" in decision: workflow_type = "CASE_1_MISSING_TABLE"
        elif "CASE_3" in decision: workflow_type = "CASE_3_CORRUPTED_DATA"
        elif "CASE_4" in decision: workflow_type = "CASE_4_MISSING_COLUMN"
        
        # Strict Fallback Logic
        if has_json and workflow_type == "CASE_2_NORMAL":
            print("[Router] Warning: LLM predicted Normal but Recovery Context exists. Forcing CASE_3_CORRUPTED_DATA.")
            workflow_type = "CASE_3_CORRUPTED_DATA"
        
        print(f"[Router] Diagnosis: {workflow_type}")
        
        return {
            "workflow_type": workflow_type,
            "initial_query_result": str(result),
            "initial_query_sql": sql,
            "has_recovery_json": has_json,
            "final_answer": "",
            "final_sql": ""
        }
    
    def worker_node(self, state: AgentState, case_type: str) -> Dict[str, Any]:
        """Worker node - handle specific case"""
        llm = get_llm("glm4.7")
        question = state['question']
        db_path = state['db_path']
        recovery_path = state['recovery_json_path']
        
        schema = get_db_schema(db_path)
        
        # Prepare context
        context = {}
        context_str = "{}"
        if state.get('has_recovery_json', False) and os.path.exists(recovery_path):
            try:
                context = read_json_file(recovery_path)
                context_str = json.dumps(context, indent=2)
                if len(context_str) > 5000:
                    if 'data_payload' in context and isinstance(context['data_payload'], list):
                        original_len = len(context['data_payload'])
                        if original_len > 10:
                            truncated_payload = context['data_payload'][:10]
                            context_for_prompt = context.copy()
                            context_for_prompt['data_payload'] = truncated_payload
                            context_for_prompt['note'] = f"Data truncated. Showing first 10 of {original_len} rows."
                            context_str = json.dumps(context_for_prompt, indent=2)
                    if len(context_str) > 10000:
                        context_str = context_str[:10000] + "... [TRUNCATED]"
            except Exception:
                context_str = "{}"

        # Use specified Prompt mode
        worker_prompt = get_prompt_template(self.prompt_mode, case_type)
        
        messages = [
            SystemMessage(content=worker_prompt),
            HumanMessage(content=f"""
            Context (Recovery JSON): {context_str}
            DB Schema: {schema}
            Question: {question}
            """)
        ]
        
        current_messages = messages.copy()
        final_res = ""
        last_run_sql = ""
        last_query_result = ""
        repair_sql_log = []
        query_executed = False
        
        for _ in range(6):
            response = llm.invoke(current_messages)
            content = response.content
            current_messages.append(response)
            
            if "FINAL ANSWER:" in content:
                if not query_executed:
                    current_messages.append(
                        HumanMessage(content="You must first provide and execute a query SQL (in ```sql``` block) before outputting FINAL ANSWER.")
                    )
                    continue
                final_res = content.split("FINAL ANSWER:")[1].strip()
                break
            
            sql_candidates = self._extract_sql_candidates(content)
            if not sql_candidates:
                continue

            for sql in sql_candidates:
                for stmt in self._split_sql_statements(sql):
                    upper_stmt = stmt.strip().upper()
                    is_query = upper_stmt.startswith("SELECT") or upper_stmt.startswith("WITH") or upper_stmt.startswith("PRAGMA")
                    if is_query:
                        print(f"[Worker] Executing Query SQL:\n{stmt}\n")
                        last_run_sql = stmt
                        result = run_sqlite_query(db_path, stmt)
                        last_query_result = str(result)
                        query_executed = True
                        current_messages.append(HumanMessage(content=f"Query Execution Result: {result}"))
                    else:
                        print(f"[Worker] Executing Repair SQL:\n{stmt}\n")
                        repair_sql_log.append(stmt)
                        result = run_sqlite_query(db_path, stmt)
                        current_messages.append(HumanMessage(content=f"Repair Execution Result: {result}"))
        
        return {
            "final_answer": final_res,
            "final_sql": last_run_sql,
            "repair_sql": ";\n".join(repair_sql_log),
            "final_query_result": last_query_result,
        }
    
    def run_single_question(self, idx: int, question: str, db_id: str, 
                           gold_sql: str, workflow_type: str,
                           db_path: str, recovery_json_path: str) -> Dict[str, Any]:
        """Run single question"""
        
        # Create temporary database
        import shutil
        temp_dir = self.output_dir / "temp_db"
        temp_dir.mkdir(exist_ok=True)
        temp_db_path = str(temp_dir / f"db_{idx:04d}.sqlite")
        shutil.copy2(db_path, temp_db_path)
        
        # Record gold SQL execution result on original database
        original_gold_result = run_sqlite_query(db_path, gold_sql)
        
        try:
            # Initialize state
            state: AgentState = {
                "question": question,
                "db_path": temp_db_path,
                "recovery_json_path": recovery_json_path,
                "initial_query_result": "",
                "initial_query_sql": "",
                "has_recovery_json": False,
                "workflow_type": "CASE_2_NORMAL",
                "final_answer": "",
                "final_sql": "",
                "repair_sql": ""
            }
            
            # 1. Router node
            router_result = self.router_node(state)
            state.update(router_result)
            
            # 2. Worker node
            case_map = {
                "CASE_1_MISSING_TABLE": "case1",
                "CASE_2_NORMAL": "case2",
                "CASE_3_CORRUPTED_DATA": "case3",
                "CASE_4_MISSING_COLUMN": "case4"
            }
            case_type = case_map.get(state['workflow_type'], "case2")
            worker_result = self.worker_node(state, case_type)
            state.update(worker_result)
            
            # 3. Evaluate results
            eval_result = {
                "eval_is_match": False,
                "eval_diff_summary": "",
                "eval_final_is_match": False,
                "eval_final_diff_summary": "",
                "eval_final_result": None
            }
            
            # Evaluate repair success rate
            if (isinstance(original_gold_result, str) and 
                original_gold_result.startswith("Error:")):
                eval_result["eval_is_match"] = False
                eval_result["eval_diff_summary"] = "INVALID_GOLD_SQL"
            else:
                repaired_gold_result = run_sqlite_query(temp_db_path, gold_sql)
                eval_result["eval_sandbox_gold_result"] = truncate_for_json(repaired_gold_result)
                
                is_match, diff_summary = compare_query_results(
                    gold_sql, original_gold_result, repaired_gold_result
                )
                eval_result["eval_is_match"] = bool(is_match)
                eval_result["eval_diff_summary"] = diff_summary
            
            # Evaluate Text-to-SQL accuracy
            if state['final_sql']:
                final_result = run_sqlite_query(temp_db_path, state['final_sql'])
                eval_result["eval_final_result"] = truncate_for_json(final_result)
                
                final_is_match, final_diff_summary = compare_query_results(
                    gold_sql, original_gold_result, final_result
                )
                eval_result["eval_final_is_match"] = bool(final_is_match)
                eval_result["eval_final_diff_summary"] = final_diff_summary
            
            record = {
                "index": idx,
                "question": question,
                "db_id": db_id,
                "workflow_type": state['workflow_type'],
                "gold_sql": gold_sql,
                "status": "completed",
                "initial_query_sql": state['initial_query_sql'],
                "initial_query_result": truncate_for_json(state['initial_query_result']),
                "repair_sql": state['repair_sql'],
                "final_sql": state['final_sql'],
                "final_answer": state['final_answer'],
                **eval_result
            }
            
        except Exception as e:
            print(f"[Error] Question {idx} execution failed: {e}")
            record = {
                "index": idx,
                "question": question,
                "db_id": db_id,
                "workflow_type": workflow_type,
                "gold_sql": gold_sql,
                "status": "error",
                "error": str(e)
            }
        finally:
            # Clean up temporary database
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)
        
        return record
    
    def run_experiment(self, start: int = 0, end: Optional[int] = None,
                      data_file: str = "bird/dev/dev_modified.json",
                      db_root: str = "bird/dev/dev_databases"):
        """Run complete experiment"""
        
        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if end is None:
            end = len(data)
        
        print(f"\nStarting Prompt engineering experiment ({self.prompt_mode}mode): Questions {start} to {end-1}")
        print(f"="*60)
        
        for idx in range(start, min(end, len(data))):
            item = data[idx]
            question = item.get('question', '')
            db_id = item.get('db_id', '')
            gold_sql = item.get('SQL', '')
            workflow_type = item.get('workflow_type', 'CASE_2_NORMAL')
            
            # Build paths
            db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
            recovery_json_path = os.path.join(db_root, db_id, "recover.json")
            
            print(f"\n{'='*60}")
            print(f"Question {idx}: {question[:50]}...")
            print(f"Workflow: {workflow_type}")
            print(f"DB: {db_id}")
            
            # Run single question
            record = self.run_single_question(
                idx=idx,
                question=question,
                db_id=db_id,
                gold_sql=gold_sql,
                workflow_type=workflow_type,
                db_path=db_path,
                recovery_json_path=recovery_json_path
            )
            
            # Save results
            self.save_result(record)
            
            # Print progress
            if (idx - start + 1) % 10 == 0:
                print(f"\nProgress: {idx - start + 1}/{end - start} questions completed")
        
        # Complete experiment
        print(f"\n{'='*60}")
        print(f"Prompt engineering experiment（{self.prompt_mode}mode) completed!")
        print(f"Results saved in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prompt Engineering Experiment")
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["zero_shot", "one_shot", "few_shot"],
                       help="Prompt mode: zero_shot, one_shot, or few_shot")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=None, help="End question index (exclusive)")
    parser.add_argument("--data-file", type=str, default="bird/dev/dev_modified.json", 
                       help="Data file path")
    parser.add_argument("--db-root", type=str, default="bird/dev/dev_databases", 
                       help="Database root directory")
    
    args = parser.parse_args()
    
    # Create experiment and run
    experiment = PromptExperiment(prompt_mode=args.mode)
    experiment.run_experiment(
        start=args.start,
        end=args.end,
        data_file=args.data_file,
        db_root=args.db_root
    )


if __name__ == "__main__":
    main()
