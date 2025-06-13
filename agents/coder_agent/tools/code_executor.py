import sys
import os
import traceback


class CodeExecutor:
    name: str = "Code Execution Tool"
    description: str = "Run Python code in a specified file and returns success or error message"

    def _run(code_file, *args):
        # Save the current sys.path for restoration after execution
        original_sys_path = sys.path.copy()

        try:
            # Add the directory of the code file to sys.path for local imports
            code_dir = os.path.dirname(os.path.abspath(code_file))
            sys.path.insert(0, code_dir)

            # Read and execute the code in the specified file
            with open(code_file, 'r') as file:
                code = file.read()

            # Set up custom globals with arguments
            exec_globals = {"__name__": "__main__", "args": args}
            exec(code, exec_globals)
            return "Success"
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            error_trace = traceback.format_exc()
            return str(error_trace)
        finally:
            # Restore sys.path to avoid side effects
            sys.path = original_sys_path


if __name__ == '__main__':
    # Path to the generated Python code file
    code_file_path = "generated_test_code.py"
    result = CodeExecutor._run(code_file=code_file_path)
    print(result)