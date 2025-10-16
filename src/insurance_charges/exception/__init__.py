# src/insurance_charges/exception/__init__.py
import os
import sys

def error_message_detail(error, error_detail: sys):
    """
    Extract detailed error message with file name and line number
    """
    try:
        _, _, exc_tb = error_detail.exc_info()
        
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            error_message = f"Error occurred in python script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
        else:
            # Fallback when no traceback is available
            error_message = f"Error occurred: [{str(error)}]"
        
        return error_message
    except Exception as e:
        # Ultimate fallback if everything fails
        return f"Error processing exception: {str(e)}, Original error: {str(error)}"

class InsuranceException(Exception):
    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        :param error_detail: error detail from sys
        """
        try:
            super().__init__(error_message)
            self.error_message = error_message_detail(error_message, error_detail=error_detail)
        except Exception as e:
            # If custom error handling fails, use basic exception
            super().__init__(error_message)
            self.error_message = str(error_message)

    def __str__(self):
        return self.error_message