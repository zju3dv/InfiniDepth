from omegaconf import DictConfig
from training.utils.logger import Log


class LRTable:
    def __init__(self,
                 default_lr: float = 1e-4,
                 prefix_table: DictConfig = None,
                 postfix_table: DictConfig = None,
                 else_table: DictConfig = None):
        # 
        self.default_lr = default_lr
        self.prefix_table = prefix_table
        self.postfix_table = postfix_table
        self.else_table = else_table
        self.tables = [self.prefix_table, self.postfix_table, self.else_table]
        self.tags = ["prefix_", "postfix_", "else_"]
        
    def match_table(self, key) -> bool:
        if self.prefix_table is not None:
            for prefix in self.prefix_table:
                if key.startswith(prefix):
                    return 0, prefix
        
        if self.postfix_table is not None:
            for postfix in self.postfix_table:
                if key.endswith(postfix):
                    return 1, postfix
        
        if self.else_table is not None:
            for else_key in self.else_table:
                if else_key in key and self.tables[2][else_key] is not None:
                    return 2, else_key
        
        return 3, None        
        
    def get_lr(self, key) -> float:
        table_idx, match_key = self.match_table(key)
        if table_idx == 3:
            Log.debug(f'{key} is not matched to any table, use default lr: {self.default_lr}')
            return 'default', self.default_lr
        else:
            Log.debug(f'{key} is matched to table {self.tags[table_idx]}: {match_key}')
            return self.tags[table_idx] + match_key, self.tables[table_idx][match_key]