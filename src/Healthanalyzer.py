import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from src.io_utils import load_file

class HealthAnalyzer:
    """
    Genomför statistika analyser och simuleringar för givet dataset.

    Hanterar statistik, konfidensintervavll, t-test, power-simuleringar
    enkel och multipel regression.
    """
    def __init__(self, path: str):
        self.df: pd.DataFrame = load_file(path)
        self.simple_model: LinearRegression
        self.multi_model: LinearRegression

    def desc_stats(self) -> pd.DataFrame:
        """
        Tar fram statistik för numeriska kolumner
        Ger oss:
        Medelvärde, median, minsta värde, högsta värde
        För kolumnerna:
        Ålder, vikt, längd, blodtryck, kolesterol
        """
        return (
            self.df[["age", "weight", "height", "systolic_bp", "cholesterol"]]
            .agg(["mean", "median", "min", "max"])
        )
    def sim_disease(self, n, column):
        """
        Simulerar en andel som har sjukdom och genom urval med återlägging.
        Ger oss andelar simulerade personer med sjukdom
        """
        sample = np.random.choice(self.df[column], size=n, replace=True)
        share_sample = np.mean(sample)
        return share_sample
        
    def ci_boot(self, n_boot, column):
        """
        Räknar fram ett 95% konfidensintervall med bootstrapmetoden
        Medelvärdet av systolic_bp
        Returnerar lägsta och högsta värdet i konfidensintervallet
        """
        mean_boot = np.empty(n_boot)

        for i in range(n_boot):
            boot_sample = np.random.choice(self.df[column], size=len(self.df), replace=True)
            mean_boot[i] = np.mean(boot_sample)

        boot_low = np.percentile(mean_boot, 2.5)
        boot_high = np.percentile(mean_boot, 97.5)

        return boot_low, boot_high

    def ci_norm(self, column):
        """
        Räknar fram ett 95% konfidensintervall med normalapproximationsmetoden
        Använder kolumnen systolic_bp
        Returnerar lägsta och högsta värdet i konfidensintervallet
        """
        normal_mean_systolic_bp = np.mean(self.df[column])

        n_normal = len(self.df)
        normal_std_systolic_bp = np.std(self.df[column], ddof=1)
        normal_se_systolic_bp = normal_std_systolic_bp / np.sqrt(n_normal)

        z = 1.96
        half_width = z * normal_se_systolic_bp

        normal_low = normal_mean_systolic_bp - half_width
        normal_high = normal_mean_systolic_bp + half_width

        return normal_low, normal_high

    def _smoker_groups(self):
        """
        Hjälpfunktion för att separera rökare och icke-rökare
        Används för hypotesprövning och power-beräkningar
        """
        smoker = self.df.loc[self.df["smoker"] == "Yes", "systolic_bp"].values
        non_smoker = self.df.loc[self.df["smoker"] == "No", "systolic_bp"].values
        return smoker, non_smoker

    def _num_smoker_groups(self):
        """
        Hjälpfunktion för att ta fram antalet rökare och icke-rökare
        Används för Power-beräkningar
        """
        smoker, non_smoker = self._smoker_groups()
        n_smoker = len(smoker)
        n_non_smoker = len(non_smoker)
        return n_smoker, n_non_smoker
    
    def _stat_smoker_groups(self):
        """
        Hjälpfunktion för att ta fram:
        Medelvärde, standardavvikelse för icke-rökare
        Standardavvikelse för rökare
        Används för Power-beräkningar.
        """
        smoker, non_smoker = self._smoker_groups()
        mean_non_smoker = np.mean(non_smoker)
        std_non_smoker = np.std(non_smoker, ddof= 1)
        std_smoker = np.std(smoker, ddof= 1)
        return mean_non_smoker, std_non_smoker,std_smoker

    def hypothesis_testing(self):
        """
        Genomför ett ensidigt welch t-test
        Ger oss t-värde och p-värde
        """
        smoker, non_smoker = self._smoker_groups()
        t_stat_w, p_value_w = stats.ttest_ind(
            smoker, non_smoker, equal_var=False, alternative="greater"
        )

        return t_stat_w, p_value_w

    def sim_hypothesis_power(self, n_sims, alpha, effect):
        """
        Gör en simulering för att få fram power-värde för en specifik effekt.
        Returnerar power-värdet
        """
        n_smoker, n_non_smoker = self._num_smoker_groups()
        mean_non_smoker, std_non_smoker, std_smoker = self._stat_smoker_groups()

        significant_test = 0

        for _ in range(n_sims):
            sim_non_smoker = np.random.normal(mean_non_smoker, std_non_smoker, n_non_smoker)
            sim_smoker = np.random.normal(mean_non_smoker + effect, std_smoker, n_smoker)

            _, p_value_w_sim = stats.ttest_ind(sim_smoker, sim_non_smoker, equal_var=False, alternative="greater")

            if p_value_w_sim < alpha:
                significant_test += 1

        power = significant_test / n_sims

        return power

    def which_power(self, n_sims, effects, alpha):
        """
        Beräknar olika power-värden för olika effekter för att kunna se vilken effekt
        som krävs för att nå över 80% power.
        Returnerar en lista med effekterna och power-värderna
        """
        n_smoker, n_non_smoker = self._num_smoker_groups()
        mean_non_smoker, std_non_smoker, std_smoker = self._stat_smoker_groups()

        results = []

        for eff in effects:
            significant = 0
            for _ in range(n_sims):
                sim_non_smoker = np.random.normal(mean_non_smoker, std_non_smoker, n_non_smoker)
                sim_smoker = np.random.normal(mean_non_smoker + eff, std_smoker, n_smoker)
                _, p = stats.ttest_ind(sim_smoker, sim_non_smoker, equal_var=False, alternative="greater")
                if p < alpha:
                    significant += 1
            power = significant / n_sims
            results.append((eff,power))
        return results

    def linear_regression(self, x_column, y_column,):
        """
        Utför beräkningar för en linjär regression.
        Ger oss:
        Skärningen, lutningen och r^2  
        """
        X= self.df[[x_column]]
        y= self.df[y_column] 

        self.simple_model = LinearRegression()
        self.simple_model.fit(X, y)

        intercept = float(self.simple_model.intercept_)
        slope = float(self.simple_model.coef_[0])
        r2 = float(self.simple_model.score(X, y))

        return intercept, slope, r2, X, y

    def predictions_and_residuals(self, X, y):
        """
        Räknar fram prognosiserade värden och residualer.
        """
        predictions = self.simple_model.predict(X)

        residuals = y - predictions
        return predictions, residuals

    def multiple_regression(self, x_column_one, x_column_two, y_column,):
        """
        Utför beräkningar för en multipel regression.
        Ger oss:
        Skärningen, lutningen för kolumn x1, lutningen för x2 och r^2 
        """
        X= self.df[[x_column_one, x_column_two]]
        y= self.df[y_column] 

        self.multi_model = LinearRegression()
        self.multi_model.fit(X, y)
    
        intercept = float(self.multi_model.intercept_)
        slope_multi_one = float(self.multi_model.coef_[0])
        slope_multi_two = float(self.multi_model.coef_[1])
        r2 = float(self.multi_model.score(X, y))
    
        return intercept, slope_multi_one, slope_multi_two, r2, X, y

    def predictions_multi(self, X):
        """
        Räknar fram prognosiserade värden
        """
        predictions_multi = self.multi_model.predict(X)
        return predictions_multi

            


