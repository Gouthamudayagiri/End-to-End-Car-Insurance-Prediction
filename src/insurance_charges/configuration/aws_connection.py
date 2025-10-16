import boto3
import os
from insurance_charges.constants import AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_ACCESS_KEY_ID_ENV_KEY, REGION_NAME

class S3Client:
    s3_client = None
    s3_resource = None
    
    def __init__(self, region_name=REGION_NAME):
        """ 
        This Class gets aws credentials from env_variable and creates an connection with s3 bucket 
        and raise exception when environment variable is not set
        """
        if S3Client.s3_resource is None or S3Client.s3_client is None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
            
            if __access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not set.")
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")
        
            S3Client.s3_resource = boto3.resource('s3',
                                            aws_access_key_id=__access_key_id,
                                            aws_secret_access_key=__secret_access_key,
                                            region_name=region_name
                                            )
            S3Client.s3_client = boto3.client('s3',
                                        aws_access_key_id=__access_key_id,
                                        aws_secret_access_key=__secret_access_key,
                                        region_name=region_name
                                        )
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
    def validate_aws_credentials(self) -> bool:
        """
        Validate AWS credentials by making a simple S3 call
        """
        try:
            # Try to list buckets to validate credentials
            response = self.s3_client.list_buckets()
            logging.info("AWS credentials validated successfully")
            return True
        except Exception as e:
            logging.error(f"AWS credential validation failed: {e}")
            return False

    def get_bucket_exists(self, bucket_name: str) -> bool:
        """
        Check if S3 bucket exists
        """
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except Exception as e:
            logging.warning(f"Bucket {bucket_name} does not exist or is inaccessible: {e}")
            return False