import os

import snowflake.connector
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas


class SnowflakeConnector:
    def __init__(self):
        self.config = {
            'user': os.getenv('SNOWFLAKE_USER', 'none'),
            'password': os.getenv('SNOWFLAKE_PASSWORD', 'none'),
            'account': os.getenv('SNOWFLAKE_ACCOUNT', 'none'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'none'),
            'database': 'PUMS',
        }

        self.ctx = snowflake.connector.connect(**self.config)
        self.cursor = self.ctx.cursor()
        print("Connected to Snowflake")

    def query(self, sql: str) -> pd.DataFrame:
        """Run a SQL query and return the result as a DataFrame."""
        try:
            df = pd.read_sql(sql, self.ctx)
            print(f"Query executed: {len(df)} rows retrieved")
            return df
        except Exception as e:
            print("Query failed:", e)
            return pd.DataFrame()

    def upload(self, df: pd.DataFrame, table_name: str, auto_create=True, overwrite=False):
        """Upload a DataFrame to a Snowflake table."""
        try:
            success, nchunks, nrows, _ = write_pandas(
                self.ctx,
                df,
                table_name=table_name,
                schema=self.config.get('schema', 'PUBLIC'),
                auto_create_table=auto_create,
                overwrite=overwrite
            )
            if success:
                print(f"Uploaded {nrows} rows to {table_name}")
            else:
                print("Upload failed")
        except Exception as e:
            print("Upload error:", e)

    def close(self):
        self.cursor.close()
        self.ctx.close()
        print("Snowflake connection closed")