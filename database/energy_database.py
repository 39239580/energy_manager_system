import logging
from database.base import Database
from database.module.users import UserModel
from database.module.data import HistoricalDataModel
from database.module.load_prediction import LoadPredictionModel
from database.module.price_prediction import PricePredictionModel
from database.module.strategy import StrategyModel
from database.module.financial import FinancialAnalysisModel
from database.module.carbon import CarbonManagementModel


logger = logging.getLogger("EnergyDatabase")


class EnergyDatabase(object):
    """能源数据库管理系统"""

    def __init__(self, db_name: str = 'energy_data.db'):
        self.db = Database(db_name)
        self.users = UserModel(self.db)
        self.historical = HistoricalDataModel(self.db)
        self.load_predictions = LoadPredictionModel(self.db)
        self.price_predictions = PricePredictionModel(self.db)
        self.strategies = StrategyModel(self.db)
        self.financial = FinancialAnalysisModel(self.db)
        self.carbon = CarbonManagementModel(self.db)  # 碳管理模块

        self.initialize_database()

    def initialize_database(self):
        """初始化所有数据表"""
        # # 避免在初始化时使用事务
        self.users.create_table()
        self.historical.create_table()
        self.load_predictions.create_table()
        self.price_predictions.create_table()
        self.strategies.create_table()
        self.financial.create_table()
        self.carbon.create_table()
        logger.info("Database tables initialized")

    def close(self):
        """关闭数据库连接"""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()