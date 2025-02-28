# import os
# import re
# from typing import List, Dict, Optional
# import logging
# import pandas as pd
# from sqlalchemy import create_engine
# from .utils import execute_sql_query

# logger = logging.getLogger(__name__)

# class PredictionQueryGenerator:
#     def __init__(self, file_info, settings=None):
#         """
#         Initialize with the PredictionFileInfo object for the uploaded dataset and optional PredictiveSettings.
#         """
#         self.file_info = file_info
#         self.settings = settings  # Optional, for user-confirmed columns
#         self.schema = self._get_schema()
#         self.table_name = self._get_table_name()

#     def _get_schema(self) -> List[Dict]:
#         """Extract schema from PredictionFileInfo."""
#         if self.file_info and self.file_info.schema:
#             return self.file_info.schema
#         logger.warning("No schema found for the dataset.")
#         return []

#     def _get_table_name(self) -> str:
#         """Extract sanitized table name from file info."""
#         if self.file_info and self.file_info.name:
#             table_name_raw = os.path.splitext(self.file_info.name)[0]
#             return self._sanitize_identifier(table_name_raw)
#         return "default_table"

#     def _sanitize_identifier(self, name: str) -> str:
#         """Sanitize identifiers for SQL compatibility."""
#         return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())

#     def _determine_columns(self) -> tuple[str, Optional[str], Optional[str], List[str]]:
#         """
#         Determine entity, time, target, and feature columns using PredictiveSettings and schema.
#         """
#         schema_columns = [col["column_name"] for col in self.schema]

#         # Use PredictiveSettings if available and valid
#         if self.settings:
#             entity_column = self.settings.entity_column if self.settings.entity_column in schema_columns else None
#             time_column = self.settings.time_column if self.settings.time_column in schema_columns and any(col["data_type"] == "timestamp" for col in self.schema) else None
#             target_column = self.settings.target_column if self.settings.target_column in schema_columns else None
#         else:
#             entity_column = time_column = target_column = None

#         # Fallback to schema analysis if settings are missing or invalid
#         if not entity_column:
#             likely_entities = [col for col in schema_columns if "id" in col.lower() or "store" in col.lower()]
#             entity_column = likely_entities[0] if likely_entities else schema_columns[0] if schema_columns else "entity"
#             logger.warning(f"No entity column in settings, defaulting to {entity_column} from schema.")

#         if not time_column and any(col["data_type"] == "timestamp" for col in self.schema):
#             time_column = next((col["column_name"] for col in self.schema if col["data_type"] == "timestamp"), None)
#             logger.warning(f"No time column in settings, defaulting to {time_column} from schema if timestamp exists.")

#         if not target_column:
#             likely_targets = [col for col in schema_columns if "sales" in col.lower() or "revenue" in col.lower()]
#             target_column = likely_targets[0] if likely_targets else None
#             logger.warning(f"No target column in settings, defaulting to {target_column} from schema if applicable.")

#         # Determine feature columns (exclude entity, time, and target)
#         feature_columns = [col for col in schema_columns if col not in [entity_column, time_column, target_column] and col != entity_column]
#         if not feature_columns:
#             feature_columns = ["1 AS placeholder"]
#             logger.warning("No feature columns identified, using placeholder.")

#         logger.info(f"Determined columns - Entity: {entity_column}, Time: {time_column}, Target: {target_column}, Features: {feature_columns}")
#         return entity_column, time_column, target_column, feature_columns

#     def generate_prediction_queries(self) -> Dict[str, str]:
#         """
#         Generate prediction-level SQL queries based on the determined columns and schema, following Pecan AI’s approach.
#         """
#         if not self.schema:
#             raise ValueError("Schema not available for the dataset.")

#         self.entity_column, self.time_column, self.target_column, self.feature_columns = self._determine_columns()

#         queries = {}

#         # 1. Sampling Query: Select entities and sampled date (today with timestamp, as in Pecan AI)
#         sampling_query = self._generate_sampling_query()
#         queries["sampling_query"] = sampling_query

#         # 2. Feature Query: Join with historical data for features (1-year lookback by default)
#         feature_query = self._generate_feature_query()
#         queries["feature_query"] = feature_query

#         return queries

#     def _generate_sampling_query(self) -> str:
#         """
#         Generate a sampling query to select entities and a sampled date (today’s timestamp, as in Pecan AI).
#         """
#         if not self.entity_column:
#             raise ValueError("Entity column could not be determined for sampling.")

#         # Use current_timestamp() exactly as in Pecan AI
#         sampled_date = "current_timestamp() AS sampled_date"

#         # Base query: Select distinct entities with today's timestamp
#         query = f"""
#             SELECT DISTINCT {self.entity_column}, {sampled_date}
#             FROM {self.table_name}
#         """

#         # Optional: Filter for entities with activity before today if time_column exists
#         if self.time_column and self._has_date_column():
#             query += f"""
#                 WHERE {self.time_column} < current_date()
#             """

#         query += " ORDER BY sampled_date DESC, {entity_column} ASC"
#         return query.format(entity_column=self.entity_column)

#     def _has_date_column(self) -> bool:
#         """Check if the schema includes any timestamp columns."""
#         return any(col["data_type"] == "timestamp" for col in self.schema)

#     def _generate_feature_query(self) -> str:
#         """
#         Generate a feature query to join historical data (1-year lookback by default) with sampled data.
#         """
#         if not self.entity_column:
#             raise ValueError("Entity column is required for features.")

#         # Identify feature columns (exclude entity, time, and target if specified)
#         feature_selects = ",\n    " + ",\n    ".join([f"tbl.{col}" for col in self.feature_columns])

#         # Base query: Join with historical data (1-year lookback)
#         query = f"""
#             WITH entity_samples AS (
#                 {self._generate_sampling_query().rstrip(';')}
#             )
#             SELECT 
#                 entity_samples.{self.entity_column},
#                 entity_samples.sampled_date,
#                 {feature_selects}
#             FROM entity_samples
#             INNER JOIN {self.table_name} AS tbl
#                 ON tbl.{self.entity_column} = entity_samples.{self.entity_column}
#         """

#         if self.time_column and self._has_date_column():
#             query += f"""
#                 AND tbl.{self.time_column} < entity_samples.sampled_date
#                 AND tbl.{self.time_column} >= date_add('year', -1, entity_samples.sampled_date)
#             """

#         query += " ORDER BY entity_samples.sampled_date DESC, entity_samples.{entity_column} ASC"
#         return query.format(entity_column=self.entity_column)

#     def execute_queries(self, aws_access_key_id, aws_secret_access_key, aws_region, athena_schema, s3_staging_dir) -> Dict[str, pd.DataFrame]:
#         """
#         Execute the generated queries and return results as DataFrames, using provided AWS credentials and settings.
#         """
#         queries = self.generate_prediction_queries()
#         results = {}
#         connection_string = (
#             f"awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}"
#             f"@athena.{aws_region}.amazonaws.com:443/{athena_schema}"
#             f"?s3_staging_dir={s3_staging_dir}&catalog_name=AwsDataCatalog"
#         )
#         engine = create_engine(connection_string)

#         for query_type, query in queries.items():
#             try:
#                 df = pd.read_sql_query(query, engine)
#                 results[query_type] = df
#                 logger.info(f"Successfully executed {query_type} query. Rows returned: {len(df)}")
#             except Exception as e:
#                 logger.error(f"Error executing {query_type} query: {e}")
#                 results[query_type] = pd.DataFrame()

#         return results

#     def validate_queries(self, aws_access_key_id, aws_secret_access_key, aws_region, athena_schema, s3_staging_dir) -> bool:
#         """
#         Validate that the generated queries are syntactically correct and align with the schema.
#         """
#         queries = self.generate_prediction_queries()
#         connection_string = (
#             f"awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}"
#             f"@athena.{aws_region}.amazonaws.com:443/{athena_schema}"
#             f"?s3_staging_dir={s3_staging_dir}&catalog_name=AwsDataCatalog"
#         )
#         engine = create_engine(connection_string)

#         for query_type, query in queries.items():
#             try:
#                 # Basic syntax check via execution (limited rows)
#                 df = pd.read_sql_query(query + " LIMIT 1", engine)
#                 if df.empty and query_type == "sampling_query":
#                     logger.warning(f"{query_type} query returned no data, but may be valid for future predictions.")
#                 elif df.empty:
#                     logger.error(f"{query_type} query returned no data and may be invalid.")
#                     return False
#             except Exception as e:
#                 logger.error(f"Validation failed for {query_type} query: {e}")
#                 return False
#         return True




import os
import re
from typing import List, Dict, Optional
import logging
import pandas as pd
from sqlalchemy import create_engine
from .utils import execute_sql_query

logger = logging.getLogger(__name__)

class PredictionQueryGenerator:
    def __init__(self, file_info, settings=None):
        """
        Initialize with the PredictionFileInfo object for the uploaded dataset and optional PredictiveSettings.
        """
        self.file_info = file_info
        self.settings = settings  # Optional, for user-confirmed columns
        self.schema = self._get_schema()
        self.table_name = self._get_table_name()

    def update_with_predictive_settings(self, predictive_settings):
        """
        Optional helper method to store the PredictiveSettings object
        if you need to call this after initialization.
        """
        self.settings = predictive_settings

    def _get_schema(self) -> List[Dict]:
        """Extract schema from PredictionFileInfo."""
        if self.file_info and self.file_info.schema:
            return self.file_info.schema
        logger.warning("No schema found for the dataset.")
        return []

    def _get_table_name(self) -> str:
        """Extract sanitized table name from file info."""
        if self.file_info and self.file_info.name:
            table_name_raw = os.path.splitext(self.file_info.name)[0]
            return self._sanitize_identifier(table_name_raw)
        return "default_table"

    def _sanitize_identifier(self, name: str) -> str:
        """Sanitize identifiers for SQL compatibility."""
        return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())

    def _determine_columns(self) -> tuple[str, Optional[str], Optional[str], List[str]]:
        """
        Determine entity, time, target, and feature columns using PredictiveSettings and schema.
        """
        schema_columns = [col["column_name"] for col in self.schema]

        # Use PredictiveSettings if available and valid
        if self.settings:
            entity_column = (self.settings.entity_column
                             if self.settings.entity_column in schema_columns else None)
            time_column = (self.settings.time_column
                           if self.settings.time_column in schema_columns
                           and any(col["data_type"] == "timestamp" for col in self.schema)
                           else None)
            target_column = (self.settings.target_column
                             if self.settings.target_column in schema_columns else None)
        else:
            entity_column = time_column = target_column = None

        # Fallback to schema analysis if settings are missing or invalid
        if not entity_column:
            likely_entities = [col for col in schema_columns if "id" in col.lower() or "store" in col.lower()]
            entity_column = (likely_entities[0] if likely_entities
                             else (schema_columns[0] if schema_columns else "entity"))
            logger.warning(f"No entity column in settings, defaulting to {entity_column} from schema.")

        if not time_column and any(col["data_type"] == "timestamp" for col in self.schema):
            time_column = next((col["column_name"] for col in self.schema if col["data_type"] == "timestamp"), None)
            logger.warning(f"No time column in settings, defaulting to {time_column} from schema if timestamp exists.")

        if not target_column:
            likely_targets = [col for col in schema_columns if "sales" in col.lower() or "revenue" in col.lower()]
            target_column = likely_targets[0] if likely_targets else None
            logger.warning(f"No target column in settings, defaulting to {target_column} from schema if applicable.")

        # Determine feature columns (exclude entity, time, and target)
        feature_columns = [
            col for col in schema_columns
            if col not in [entity_column, time_column, target_column]
        ]
        if not feature_columns:
            feature_columns = ["1 AS placeholder"]
            logger.warning("No feature columns identified, using placeholder.")

        logger.info(f"Determined columns - Entity: {entity_column}, Time: {time_column}, "
                    f"Target: {target_column}, Features: {feature_columns}")
        return entity_column, time_column, target_column, feature_columns

    def generate_prediction_queries(self) -> Dict[str, str]:
        """
        Generate prediction-level SQL queries based on the determined columns and schema,
        following Pecan AI’s approach.
        """
        if not self.schema:
            raise ValueError("Schema not available for the dataset.")

        # Pull out the columns we determined
        self.entity_column, self.time_column, self.target_column, self.feature_columns = self._determine_columns()

        queries = {}

        # 1. Sampling Query: Select entities and today's timestamp
        queries["sampling_query"] = self._generate_sampling_query()

        # 2. Feature Query: Join with historical data for features
        queries["feature_query"] = self._generate_feature_query()

        return queries

    def _generate_sampling_query(self) -> str:
        """
        Generate a sampling query to select entities and a sampled date
        (today’s timestamp, cast to a timestamp w/o time zone).
        """
        if not self.entity_column:
            raise ValueError("Entity column could not be determined for sampling.")

        # Remove time zone from current_timestamp
        sampled_date = "CAST(current_timestamp AS timestamp) AS sampled_date"

        query = f"""
            SELECT DISTINCT {self.entity_column}, {sampled_date}
            FROM {self.table_name}
        """
        return query.strip()

    def _has_date_column(self) -> bool:
        """Check if the schema includes any timestamp columns."""
        return any(col["data_type"] == "timestamp" for col in self.schema)

    def _generate_feature_query(self) -> str:
        """
        Generate a feature query to join historical data (1-year lookback by default)
        with the sampling query's results.
        """
        if not self.entity_column:
            raise ValueError("Entity column is required for features.")

        feature_selects = ",\n    " + ",\n    ".join([f"tbl.{col}" for col in self.feature_columns])

        query = f"""
            WITH entity_samples AS (
                {self._generate_sampling_query().rstrip(';')}
            )
            SELECT 
                entity_samples.{self.entity_column},
                entity_samples.sampled_date
                {feature_selects}
            FROM entity_samples
            INNER JOIN {self.table_name} AS tbl
                ON tbl.{self.entity_column} = entity_samples.{self.entity_column}
        """
        if self.time_column and self._has_date_column():
            query += f"""
                AND tbl.{self.time_column} < entity_samples.sampled_date
                AND tbl.{self.time_column} >= date_add('year', -1, entity_samples.sampled_date)
            """
        return query.strip()

    def execute_queries(self,
                        aws_access_key_id,
                        aws_secret_access_key,
                        aws_region,
                        athena_schema,
                        s3_staging_dir) -> Dict[str, pd.DataFrame]:
        """
        Execute the generated queries via Athena and return results as DataFrames.
        """
        queries = self.generate_prediction_queries()
        results = {}
        connection_string = (
            f"awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}"
            f"@athena.{aws_region}.amazonaws.com:443/{athena_schema}"
            f"?s3_staging_dir={s3_staging_dir}&catalog_name=AwsDataCatalog"
        )
        engine = create_engine(connection_string)

        for query_type, query in queries.items():
            try:
                df = pd.read_sql_query(query, engine)
                results[query_type] = df
                logger.info(f"Successfully executed {query_type} query. Rows returned: {len(df)}")
            except Exception as e:
                logger.error(f"Error executing {query_type} query: {e}")
                results[query_type] = pd.DataFrame()

        return results

    def validate_queries(self,
                         aws_access_key_id,
                         aws_secret_access_key,
                         aws_region,
                         athena_schema,
                         s3_staging_dir) -> bool:
        """
        Validate that the generated queries are syntactically correct and align with the schema.
        We do this by running each query with a LIMIT 1 to see if it errors out.
        """
        queries = self.generate_prediction_queries()
        connection_string = (
            f"awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}"
            f"@athena.{aws_region}.amazonaws.com:443/{athena_schema}"
            f"?s3_staging_dir={s3_staging_dir}&catalog_name=AwsDataCatalog"
        )
        engine = create_engine(connection_string)

        for query_type, query in queries.items():
            try:
                df = pd.read_sql_query(query + " LIMIT 1", engine)
                if df.empty and query_type == "sampling_query":
                    logger.warning(
                        f"{query_type} query returned no data, but may be valid for future predictions."
                    )
                elif df.empty:
                    logger.error(f"{query_type} query returned no data and may be invalid.")
                    return False
            except Exception as e:
                logger.error(f"Validation failed for {query_type} query: {e}")
                return False

        return True
