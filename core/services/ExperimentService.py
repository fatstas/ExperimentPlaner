import json
import numpy as np
import pandas as pd
from datetime import datetime


class ExperimentService:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def get_experiments_list(self):
        """Получение списка экспериментов из БД"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM experiments ORDER BY created_date DESC")
        experiments = [exp[0] for exp in cursor.fetchall()]
        conn.close()
        return experiments

    def load_experiment(self, name):
        """Загрузка эксперимента по имени"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT name, description, parameters, adaptive_mode, target_type 
            FROM experiments WHERE name = ?
        ''', (name,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'name': result[0],
                'description': result[1] or "",
                'parameters': json.loads(result[2]),
                'adaptive_mode': bool(result[3]),
                'target_type': result[4] or 'maximize'
            }
        return None

    def check_experiment_exists(self, name):
        """Проверка существования эксперимента"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM experiments WHERE name = ?", (name,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def save_experiment(self, name, parameters, description="", adaptive_mode=False, target_type="maximize"):
        """Сохранение эксперимента"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        try:
            # Проверяем существование
            cursor.execute("SELECT id FROM experiments WHERE name = ?", (name,))
            existing = cursor.fetchone()

            if existing:
                # Обновляем существующий
                cursor.execute('''
                    UPDATE experiments 
                    SET parameters = ?, description = ?, adaptive_mode = ?, target_type = ?
                    WHERE name = ?
                ''', (json.dumps(parameters), description, adaptive_mode, target_type, name))
            else:
                # Создаем новый
                from datetime import datetime
                cursor.execute('''
                    INSERT INTO experiments (name, created_date, parameters, description, adaptive_mode, target_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                name, datetime.now().isoformat(), json.dumps(parameters), description, adaptive_mode, target_type))

            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving experiment: {e}")
            return False
        finally:
            conn.close()

    def save_experiment_results(self, experiment_name, design):
        """Сохранение результатов эксперимента"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        try:
            # Получаем ID эксперимента
            cursor.execute("SELECT id FROM experiments WHERE name = ?", (experiment_name,))
            result = cursor.fetchone()
            if not result:
                return False

            experiment_id = result[0]

            # Удаляем старые результаты
            cursor.execute("DELETE FROM experiment_results WHERE experiment_id = ?", (experiment_id,))

            # Сохраняем новый план
            for i, run in enumerate(design, 1):
                cursor.execute('''
                    INSERT INTO experiment_results (experiment_id, run_order, parameters_values)
                    VALUES (?, ?, ?)
                ''', (experiment_id, i, json.dumps(run)))

            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
        finally:
            conn.close()

    def get_experiment_results(self, experiment_name):
        """Получение результатов эксперимента"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT er.run_order, er.parameters_values, er.results, er.notes
            FROM experiment_results er
            JOIN experiments e ON e.id = er.experiment_id
            WHERE e.name = ?
            ORDER BY er.run_order
        ''', (experiment_name,))

        results = cursor.fetchall()
        conn.close()
        return results

    def get_max_run_order(self, experiment_name):
        """Получение максимального номера опыта"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT MAX(run_order) FROM experiment_results 
            WHERE experiment_id = (SELECT id FROM experiments WHERE name = ?)
        ''', (experiment_name,))

        max_order = cursor.fetchone()[0] or 0
        conn.close()
        return max_order

    def get_parameter_bounds(self, params_tree):
        """Получение границ параметров из treeview"""
        bounds = []
        for item in params_tree.get_children():
            values = params_tree.item(item, 'values')
            if len(values) == 3:
                bounds.append((float(values[1]), float(values[2])))
        return bounds

    def get_parameter_names(self, params_tree):
        """Получение списка имен параметров"""
        param_names = []
        for item in params_tree.get_children():
            values = params_tree.item(item, 'values')
            if len(values) >= 1:
                param_names.append(values[0])
        return param_names

    def get_existing_points(self, experiment_name, param_names):
        """Получение списка уже проведенных опытов"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT parameters_values FROM experiment_results 
            WHERE experiment_id = (SELECT id FROM experiments WHERE name = ?)
        ''', (experiment_name,))

        results = cursor.fetchall()
        existing_points = []

        for (params_json,) in results:
            try:
                params = json.loads(params_json)
                point = [params[param] for param in param_names]
                existing_points.append(point)
            except:
                continue

        conn.close()
        return existing_points

    def has_completed_experiments(self, experiment_name):
        """Проверка наличия завершенных опытов с результатами"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM experiment_results er
            JOIN experiments e ON e.id = er.experiment_id
            WHERE e.name = ? AND er.results IS NOT NULL AND er.results != ''
        ''', (experiment_name,))

        count = cursor.fetchone()[0] > 0
        conn.close()
        return count

    def get_training_data(self, experiment_name):
        """Получение данных для обучения модели"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT er.parameters_values, er.results
            FROM experiment_results er
            JOIN experiments e ON e.id = er.experiment_id
            WHERE e.name = ? AND er.results IS NOT NULL AND er.results != ''
        ''', (experiment_name,))

        results = cursor.fetchall()
        conn.close()

        X = []
        y = []
        param_names = []

        for params_json, result in results:
            try:
                params = json.loads(params_json)
                # Сохраняем порядок параметров при первом успешном парсинге
                if not param_names:
                    param_names = list(params.keys())

                x_vector = [params[param] for param in param_names]
                X.append(x_vector)
                y.append(float(result))
            except (json.JSONDecodeError, ValueError):
                continue

        import numpy as np
        return np.array(X), np.array(y), param_names

    def export_to_csv(self, experiment_name, results_tree):
        """Экспорт результатов в CSV"""
        try:
            # Собираем данные из таблицы
            data = []
            columns = results_tree['columns']

            # Заголовки
            headers = [results_tree.heading(col)['text'] for col in columns]
            data.append(headers)

            # Данные
            for item in results_tree.get_children():
                values = results_tree.item(item, 'values')
                data.append(values)

            # Создаем DataFrame и сохраняем
            import pandas as pd
            df = pd.DataFrame(data[1:], columns=data[0])
            return df
        except Exception as e:
            raise Exception(f"Ошибка при экспорте: {str(e)}")
