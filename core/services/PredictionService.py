import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


class PredictionService:
    def __init__(self):
        self.current_model = None
        self.scaler = None
        self.param_names = []

    def update_model(self, X, y):
        """Обновление модели машинного обучения"""
        if len(X) < 3:
            raise ValueError("Недостаточно данных для построения модели. Нужно минимум 3 опыта.")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.current_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.current_model.fit(X_scaled, y)

        return self.current_model.score(X_scaled, y)

    def predict_result(self, parameters):
        """Предсказание результата для заданных параметров"""
        if not self.current_model:
            raise ValueError("Модель не обучена.")

        X = np.array([list(parameters.values())])
        X_scaled = self.scaler.transform(X)
        return self.current_model.predict(X_scaled)[0]

    def get_feature_importance(self):
        """Получение важности признаков"""
        if self.current_model and hasattr(self.current_model, 'feature_importances_'):
            return self.current_model.feature_importances_
        return None

    def suggest_next_experiment(self, bounds, existing_points, target_type, target_value=0):
        """Предложение следующего опыта на основе модели"""
        if not self.current_model:
            raise ValueError("Модель не обучена. Сначала обновите модель.")

        def objective(x):
            x_scaled = self.scaler.transform([x])
            prediction = self.current_model.predict(x_scaled)[0]

            # Штраф за близость к существующим точкам
            penalty = 0
            for existing_point in existing_points:
                distance = np.linalg.norm(np.array(x) - np.array(existing_point))
                if distance < 0.1:
                    penalty += 100 * (0.1 - distance)

            if target_type == 'maximize':
                return -prediction + penalty
            elif target_type == 'minimize':
                return prediction + penalty
            else:  # target
                return abs(prediction - target_value) + penalty

        # Пробуем несколько случайных начальных точек
        best_result = None
        best_score = float('inf')

        for attempt in range(10):
            x0 = [np.random.uniform(low, high) for low, high in bounds]

            constraints = []
            for i, (low, high) in enumerate(bounds):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, l=low: x[i] - l})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, h=high: h - x[i]})

            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success and result.fun < best_score:
                best_score = result.fun
                best_result = result

        if best_result and best_result.success:
            suggested_params = best_result.x

            # Проверяем, что точка достаточно отличается от существующих
            min_distance = min([np.linalg.norm(np.array(suggested_params) - np.array(point))
                                for point in existing_points]) if existing_points else float('inf')

            if min_distance < 0.05:
                raise ValueError("Не найдено существенно новых оптимальных параметров.")

            return suggested_params
        else:
            raise ValueError("Не удалось найти оптимальные параметры.")

    def predict_result(self, parameters):
        """Предсказание результата для заданных параметров"""
        if not self.current_model:
            raise ValueError("Модель не обучена.")

        X = np.array([list(parameters.values())])
        X_scaled = self.scaler.transform(X)
        return self.current_model.predict(X_scaled)[0]

    def get_feature_importance(self):
        """Получение важности признаков"""
        if self.current_model and hasattr(self.current_model, 'feature_importances_'):
            return self.current_model.feature_importances_
        return None