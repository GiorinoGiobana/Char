import json
from collections import defaultdict
import os

def load_results(file_path):
    """Load results file"""
    results = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return results
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

def main():
    # Define all result files (chronological order)
    result_files = [
        "results/exp1_full_glm_20260310_225426/results.jsonl",  # Questions 1481-1533 (latest)
        "results/exp1_full_glm_20260310_010117/results.jsonl",  # Questions 1256-1391
        "results/exp1_full_glm_20260309_181712/results.jsonl",  # Questions 1214-1255
        "results/exp1_full_glm_20260309_103019/results.jsonl",  # Questions 1192-1213
        "results/exp1_full_glm_20260308_165510/results.jsonl",  # Questions 898-1191
        "results/exp1_full_glm_20260308_000848/results.jsonl",  # Questions 450-897
        "results/exp1_full_glm_20260307_002100/results.jsonl",  # Questions 0-451
    ]
    
    # Load all results
    all_results = []
    for file_path in result_files:
        print(f"Loading file: {file_path}")
        results = load_results(file_path)
        print(f"  Found {len(results)} results")
        all_results.extend(results)
    
    # Deduplicate (keep latest results)
    results_dict = {}
    for result in all_results:
        index = result['index']
        if index not in results_dict:
            results_dict[index] = result
        else:
            # If duplicates exist, keep the latter result (based on filename)
            results_dict[index] = result
    
    # Count all 1534 questions
    glm_results_full = [r for r in results_dict.values() if r['index'] < 1534]
    
    print(f'\nTotal 1534 questions count: {len(glm_results_full)}')
    
    # Detailed statistics for GLM full 1534 questions
    glm_stats = {
        'total': len(glm_results_full),
        'completed': 0,
        'timeout': 0,
        'error': 0,
        'repair_success': 0,
        'query_success': 0,
        'workflow_types': defaultdict(int),
        'by_status': defaultdict(int),
        'repair_success_by_workflow': defaultdict(int),
        'query_success_by_workflow': defaultdict(int),
        'completed_by_workflow': defaultdict(int)
    }
    
    for result in glm_results_full:
        status = result.get('status', '')
        workflow_type = result.get('workflow_type', '')
        
        glm_stats['by_status'][status] += 1
        
        if status == 'completed':
            glm_stats['completed'] += 1
            glm_stats['completed_by_workflow'][workflow_type] += 1
            
            if result.get('eval_is_match', False):
                glm_stats['repair_success'] += 1
                glm_stats['repair_success_by_workflow'][workflow_type] += 1
            
            if result.get('eval_final_is_match', False):
                glm_stats['query_success'] += 1
                glm_stats['query_success_by_workflow'][workflow_type] += 1
        
        elif status == 'timeout':
            glm_stats['timeout'] += 1
        
        elif status == 'error':
            glm_stats['error'] += 1
        
        if workflow_type:
            glm_stats['workflow_types'][workflow_type] += 1
    
    # Print statistics
    print('\n' + '='*80)
    print('GLM4.7 Full 1534 Questions Statistics')
    print('='*80)
    
    print(f'\nBasic statistics:')
    print(f'  Total questions: {glm_stats["total"]}')
    print(f'  Completed questions: {glm_stats["completed"]} ({glm_stats["completed"]/glm_stats["total"]*100:.1f}%)')
    print(f'  Timeout questions: {glm_stats["timeout"]} ({glm_stats["timeout"]/glm_stats["total"]*100:.1f}%)')
    print(f'  Error questions: {glm_stats["error"]} ({glm_stats["error"]/glm_stats["total"]*100:.1f}%)')
    
    print(f'\nRepair capability:')
    print(f'  Repair success count: {glm_stats["repair_success"]} ({glm_stats["repair_success"]/glm_stats["completed"]*100:.1f}% of completed)')
    print(f'  Repair success rate: {glm_stats["repair_success"]/glm_stats["total"]*100:.1f}%')
    
    print(f'\nText-to-SQL capability:')
    print(f'  Query success count: {glm_stats["query_success"]} ({glm_stats["query_success"]/glm_stats["completed"]*100:.1f}% of completed)')
    print(f'  Text-to-SQL accuracy: {glm_stats["query_success"]/glm_stats["total"]*100:.1f}%')
    
    print(f'\nDistribution by status:')
    for status, count in sorted(glm_stats['by_status'].items()):
        print(f'  {status}: {count} ({count/glm_stats["total"]*100:.1f}%)')
    
    print(f'\nDistribution by workflow type:')
    for workflow_type, count in sorted(glm_stats['workflow_types'].items()):
        print(f'  {workflow_type}: {count}')
    
    print(f'\nRepair success rate by workflow type:')
    for workflow_type in ['CASE_1_MISSING_TABLE', 'CASE_2_NORMAL', 'CASE_3_CORRUPTED_DATA', 'CASE_4_MISSING_COLUMN']:
        completed = glm_stats['completed_by_workflow'].get(workflow_type, 0)
        repair_success = glm_stats['repair_success_by_workflow'].get(workflow_type, 0)
        if completed > 0:
            print(f'  {workflow_type}: {repair_success}/{completed} ({repair_success/completed*100:.1f}%)')
    
    print(f'\nText-to-SQL accuracy by workflow type:')
    for workflow_type in ['CASE_1_MISSING_TABLE', 'CASE_2_NORMAL', 'CASE_3_CORRUPTED_DATA', 'CASE_4_MISSING_COLUMN']:
        completed = glm_stats['completed_by_workflow'].get(workflow_type, 0)
        query_success = glm_stats['query_success_by_workflow'].get(workflow_type, 0)
        if completed > 0:
            print(f'  {workflow_type}: {query_success}/{completed} ({query_success/completed*100:.1f}%)')
    
    # Save merged results
    merged_output_file = 'results/exp1_full_glm_merged_full/results.jsonl'
    os.makedirs(os.path.dirname(merged_output_file), exist_ok=True)
    
    with open(merged_output_file, 'w', encoding='utf-8') as f:
        for result in glm_results_full:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f'\nMerged results saved to: {merged_output_file}')
    
    # Save statistics
    output_file = 'glm_full_1534_stats.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_questions': glm_stats['total'],
            'completed': glm_stats['completed'],
            'timeout': glm_stats['timeout'],
            'error': glm_stats['error'],
            'repair_success': glm_stats['repair_success'],
            'query_success': glm_stats['query_success'],
            'completion_rate': glm_stats['completed']/glm_stats['total']*100,
            'repair_success_rate': glm_stats['repair_success']/glm_stats['total']*100,
            'query_success_rate': glm_stats['query_success']/glm_stats['total']*100,
            'workflow_types': dict(glm_stats['workflow_types']),
            'by_status': dict(glm_stats['by_status']),
            'repair_success_by_workflow': {k: v for k, v in glm_stats['repair_success_by_workflow'].items() if v > 0},
            'query_success_by_workflow': {k: v for k, v in glm_stats['query_success_by_workflow'].items() if v > 0},
            'completed_by_workflow': {k: v for k, v in glm_stats['completed_by_workflow'].items() if v > 0}
        }, f, indent=2, ensure_ascii=False)
    
    print(f'Statistics saved to: {output_file}')

if __name__ == '__main__':
    main()