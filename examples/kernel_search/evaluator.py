"""
Evaluator for kernel-search with improved timeout handling
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle

class TimeoutError(Exception):
    pass

def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        multi_ops, num_correct tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Add function run_mat_multi to the program
    if not hasattr(program, 'run_mat_multi'):
        def run_mat_multi():
            test_cases = 100
            # Generate test cases
            import numpy as np
            rng = np.random.default_rng()
            A_matrices = rng.integers(-100, 101, size=(test_cases, 3, 3)).tolist()
            B_matrices = rng.integers(-100, 101, size=(test_cases, 3, 3)).tolist()
            test_matrices = list(zip(A_matrices, B_matrices))
            
            # Verify correctness against naive implementation
            def naive_multiply(A, B):
                A_np = np.array(A)
                B_np = np.array(B)
                C_np = np.matmul(A_np, B_np)
                C = C_np.tolist()
                return C
            # Test the evolved algorithm
            #start_time = time.time()
            
            num_correct = 0
            for A, B in test_matrices:
                result = program.matmul(A, B)
                expected = naive_multiply(A, B)
                if result != expected:
                    break
                num_correct += 1
            
            #execution_time = (time.time() - start_time)/test_cases
            result, score = count_multiplications_recursive(program.matmul, A_matrices[0], B_matrices[0])
                        
            # Return score (higher is better)
            return -score, num_correct
        program.run_mat_multi = run_mat_multi
    
    # Add function count_multiplications_recursive to the program
    if not hasattr(program, 'count_multiplications_recursive'):
        def count_multiplications_recursive(func, *args, **kwargs):
            import dis, types
            total_mults = 0

            # Count in the main function
            bytecode = dis.Bytecode(func)
            for instr in bytecode:
                # Check for various loop constructs in bytecode
                if instr.opname in ['FOR_ITER', 'SETUP_LOOP', 'GET_ITER', 'GET_ANEXT', 'SETUP_ASYNC_WITH']:
                    raise Exception(f"Loop detected in the implementation {{instr.opname}}")
                    
                # Check for list/set/dict comprehensions
                if instr.opname in ['LIST_APPEND', 'SET_ADD', 'MAP_ADD']:
                    raise Exception(f"Comprehension detected in the implementation {{instr.opname}}")
                    
                # Check for while loops and conditional jumps
                if instr.opname in ['JUMP_ABSOLUTE', 'POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE', 
                                'JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP']:
                    raise Exception(f"Potential while loop or conditional jump detected in the implementation {{instr.opname}}")
                    
                # Check for generator expressions
                if instr.opname in ['YIELD_VALUE', 'YIELD_FROM']:
                    raise Exception(f"Generator expression detected in the implementation {{instr.opname}}    ")

                # Check for calls to functions that could implement loops
                if instr.opname in ['CALL_FUNCTION', 'CALL_METHOD', 'CALL_FUNCTION_KW', 'CALL_FUNCTION_EX']:
                    func_name = None
                    if hasattr(instr, 'argval') and isinstance(instr.argval, str):
                        func_name = instr.argval
                    elif hasattr(func, '__code__') and hasattr(func.__code__, 'co_names'):
                        # Try to get the name from the previous instruction
                        idx = list(bytecode).index(instr)
                        if idx > 0:
                            prev = list(bytecode)[idx-1]
                            if hasattr(prev, 'argval') and isinstance(prev.argval, str):
                                func_name = prev.argval
                    
                    loop_funcs = ['map', 'filter', 'reduce', 'sum', 'any', 'all', 'enumerate', 'zip', 
                                'sorted', 'min', 'max', 'reversed', 'join', 'split', 'replace']
                    if func_name and any(loop_func == func_name for loop_func in loop_funcs):
                        raise Exception(f"Hidden loop construct detected: {{func_name}}")
                        
                # Check for string operations that might contain implicit loops
                if instr.opname == 'LOAD_ATTR':
                    string_loop_methods = ['join', 'split', 'replace', 'strip', 'find', 'index', 'count']
                    if instr.argval in string_loop_methods:
                        raise Exception(f"String method with implicit loop detected: {{instr.argval}}")
                        
                # Check for recursive calls (potential tail recursion)
                if instr.opname == 'LOAD_GLOBAL' and instr.argval == func.__name__:
                    next_instrs = list(bytecode)[list(bytecode).index(instr):]
                    for next_instr in next_instrs[:5]:  # Check next few instructions
                        if next_instr.opname in ['CALL_FUNCTION', 'CALL_METHOD']:
                            raise Exception("Recursive call detected (implicit loop)")
                if instr.opname == 'BINARY_MULTIPLY' or (instr.opname == 'BINARY_OP' and instr.arg == 5):
                    total_mults += 1
            
            # Find and count in inner functions
            for const in func.__code__.co_consts:
                if isinstance(const, types.CodeType):
                    # This is an inner function
                    inner_func = types.FunctionType(const, func.__globals__)
                    inner_mults = count_multiplications_recursive(inner_func)
                    total_mults += inner_mults
            
            # If this is the top-level call, execute the function
            if args or kwargs:
                result = func(*args, **kwargs)
                return result, total_mults
            
            return total_mults
        program.count_multiplications_recursive = count_multiplications_recursive
    
    # Run the packing function
    print("Calling run_packing()...")
    multi_ops, num_correct = program.run_mat_multi()
    print(f"run_mat_multi() returned successfully: multi_ops = {{multi_ops}}, num_correct = {{num_correct}}")

    # Save results to a file
    results = {{
        'multi_ops': multi_ops,
        'num_correct': num_correct
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results["multi_ops"], results['num_correct']
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
           os.unlink(temp_file_path)
        if os.path.exists(results_path):
           os.unlink(results_path)

def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the times of multiplications in mm

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Target value from the paper
    TARGET_VALUE = -23 # AlphaEvolve result for 3*3*3

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        multi_ops, num_correct = run_with_timeout(
            program_path, timeout_seconds=600  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Target ratio (how close we are to the target)
        target_ratio = - multi_ops / TARGET_VALUE

        # Combined score - higher is better
        combined_score = multi_ops + num_correct

        print(
            f"Evaluation: multi_ops={multi_ops}, num_correct={num_correct}, target={TARGET_VALUE}, ratio={target_ratio:.2f}, time={eval_time:.2f}s"
        )

        return {
            "multi_ops": float(multi_ops),
            "num_correct": float(num_correct),
            "target_ratio": float(target_ratio),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "multi_ops": -27,
            "num_correct" : 0,
            "target_ratio": -1,
            "eval_time": 0.0,
            "combined_score": -27 + 100,
        }