import sqlite3
import logging
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional, Union, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Database")


class Database:
    """通用数据库操作层 - 提供基础的CRUD和事务管理"""

    def __init__(self, db_name: str = "energy_data.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor = self.conn.cursor()
        self.transaction_stack = []  # 使用栈管理事务嵌套
        logger.info(f"Database initialized: {db_name}")

    def execute(self, sql: str, params: Union[tuple, dict, None] = None) -> 'Database':
        """执行SQL语句（支持参数化查询）"""
        try:
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
        except sqlite3.Error as e:
            if self.transaction_stack:
                self.rollback()
            logger.error(f"SQL execution failed: {str(e)}")
            raise e
        return self

    def executemany(self, sql: str, params_list: List[Union[tuple, dict]]) -> 'Database':
        """批量执行SQL语句"""
        try:
            self.cursor.executemany(sql, params_list)
        except sqlite3.Error as e:
            if self.transaction_stack:
                self.rollback()
            logger.error(f"Batch execution failed: {str(e)}")
            raise e
        return self

    def fetchall(self) -> List[sqlite3.Row]:
        """获取所有查询结果（字典格式）"""
        return self.cursor.fetchall()

    def fetchone(self) -> Optional[sqlite3.Row]:
        """获取单条查询结果"""
        return self.cursor.fetchone()

    def begin_transaction(self) -> None:
        """开始事务（支持嵌套）"""
        if not self.transaction_stack:
            self.execute("BEGIN")
            logger.debug("Transaction started")
        self.transaction_stack.append(True)

    def commit(self) -> None:
        """提交事务（仅在最外层提交）"""
        if not self.transaction_stack:
            return

        self.transaction_stack.pop()
        if not self.transaction_stack:
            try:
                self.conn.commit()
                logger.debug("Transaction committed")
            except sqlite3.Error as e:
                logger.error(f"Commit failed: {str(e)}")
                raise

    def rollback(self) -> None:
        """回滚事务（回滚所有嵌套事务）"""
        if not self.transaction_stack:
            return

        self.transaction_stack = []  # 清空事务栈
        try:
            self.conn.rollback()
            logger.debug("Transaction rolled back")
        except sqlite3.Error as e:
            logger.error(f"Rollback failed: {str(e)}")
            raise

    @contextmanager
    def transaction(self):
        """事务上下文管理器（支持嵌套）"""
        # 检查是否已在事务中
        already_in_transaction = bool(self.transaction_stack)

        if not already_in_transaction:
            self.begin_transaction()

        try:
            yield
            if not already_in_transaction:
                self.commit()
        except Exception as e:
            if not already_in_transaction:
                self.rollback()
            logger.error(f"Transaction failed: {str(e)}")
            raise

    def insert(self, table: str, data: dict) -> int:
        """插入单条数据，返回rowid"""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.execute(sql, tuple(data.values()))
        return self.cursor.lastrowid

    def bulk_insert(self, table: str, columns: List[str], data: List[tuple]) -> int:
        """批量插入数据，返回插入行数"""
        placeholders = ', '.join(['?'] * len(columns))
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        self.executemany(sql, data)
        return self.cursor.rowcount

    def update(self, table: str, data: dict, condition: str, params: tuple = ()) -> int:
        """更新数据，返回影响行数"""
        set_clause = ', '.join([f"{key} = ?" for key in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        self.execute(sql, tuple(data.values()) + params)
        return self.cursor.rowcount

    def delete(self, table: str, condition: str, params: tuple = ()) -> int:
        """删除数据，返回影响行数"""
        sql = f"DELETE FROM {table} WHERE {condition}"
        self.execute(sql, params)
        return self.cursor.rowcount

    def select(self, table: str, columns: List[str] = ["*"],
               condition: str = "", params: tuple = (),
               order: str = "", limit: int = 0) -> List[sqlite3.Row]:
        """查询数据"""
        cols = ', '.join(columns)
        where = f"WHERE {condition}" if condition else ""
        order_by = order if order else ""
        limit_clause = f"LIMIT {limit}" if limit else ""
        sql = f"SELECT {cols} FROM {table} {where} {order_by} {limit_clause}"
        self.execute(sql, params)
        return self.fetchall()

    def create_table(self, table: str, schema: dict) -> None:
        """创建数据表"""
        columns = ', '.join([f"{col} {dtype}" for col, dtype in schema.items()])
        sql = f"CREATE TABLE IF NOT EXISTS {table} ({columns})"
        self.execute(sql)
        logger.info(f"Table created/verified: {table}")

    def drop_table(self, table: str) -> None:
        """删除数据表"""
        self.execute(f"DROP TABLE IF EXISTS {table}")
        logger.info(f"Table dropped: {table}")

    def create_index(self, table: str, index_name: str, columns: List[str], unique: bool = False) -> None:
        """创建索引"""
        unique_clause = "UNIQUE" if unique else ""
        sql = f"CREATE {unique_clause} INDEX IF NOT EXISTS {index_name} ON {table} ({', '.join(columns)})"
        self.execute(sql)

    def __enter__(self) -> 'Database':
        """支持with上下文管理，自动开始事务"""
        self.begin_transaction()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出上下文：无异常则提交，有异常则回滚"""
        if exc_type:
            self.rollback()
        else:
            self.commit()

    def close(self) -> None:
        """关闭数据库连接"""
        # 检查是否有未提交的事务
        if self.transaction_stack:
            self.rollback()
        self.conn.close()
        logger.info("Database connection closed")