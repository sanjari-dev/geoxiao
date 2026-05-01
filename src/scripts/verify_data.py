# src/scripts/verify_data.py
import argparse
import clickhouse_connect
import structlog
from src.config.settings import settings

log = structlog.get_logger(__name__)

def verify_clickhouse_data():
    parser = argparse.ArgumentParser(description='Geoxiao Read-Only Data Verifier')
    parser.add_argument('--symbol', type=str, default=settings.SYMBOL)
    args = parser.parse_args()

    try:
        client = clickhouse_connect.get_client(
            host=settings.CH_HOST,
            port=settings.CH_PORT,
            database=settings.CH_DATABASE,
            username=settings.CH_USER,
            password=settings.CH_PASSWORD
        )
        
        query = f"""
            SELECT 
                min(timestamp) as first_tick,
                max(timestamp) as last_tick,
                count(*) as total_rows
            FROM ticks
            WHERE instrument = '{args.symbol}'
        """
        result = client.query(query)
        row = result.result_rows[0]
        
        log.info('✅ ClickHouse Data Verification Success', 
                 symbol=args.symbol,
                 first_tick=str(row[0]),
                 last_tick=str(row[1]),
                 total_rows=row[2],
                 database=settings.CH_DATABASE,
                 table='ticks',
                 mode='READ-ONLY')
                 
    except Exception as e:
        log.error('❌ Failed to verify ClickHouse data', error=str(e))

if __name__ == '__main__':
    verify_clickhouse_data()
