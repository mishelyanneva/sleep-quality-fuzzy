import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_fuzzy_system():
    # Входни променливи
    sleep_duration = ctrl.Antecedent(np.arange(0, 13, 1), "sleep_duration")
    noise = ctrl.Antecedent(np.arange(0, 101, 1), "noise")
    light = ctrl.Antecedent(np.arange(0, 101, 1), "light")

    # Изходна променлива
    sleep_quality = ctrl.Consequent(np.arange(0, 101, 1), "sleep_quality")

    # Membership функции за продължителност на съня
    sleep_duration["short"] = fuzz.trimf(sleep_duration.universe, [0, 0, 5])
    sleep_duration["normal"] = fuzz.trimf(sleep_duration.universe, [4, 7, 9])
    sleep_duration["long"] = fuzz.trimf(sleep_duration.universe, [8, 12, 12])

    # Membership функции за шум
    noise["low"] = fuzz.trimf(noise.universe, [0, 0, 35])
    noise["medium"] = fuzz.trimf(noise.universe, [20, 50, 80])
    noise["high"] = fuzz.trimf(noise.universe, [60, 100, 100])

    # Membership функции за светлина
    light["dark"] = fuzz.trimf(light.universe, [0, 0, 30])
    light["moderate"] = fuzz.trimf(light.universe, [20, 50, 80])
    light["bright"] = fuzz.trimf(light.universe, [60, 100, 100])

    # Membership функции за качество на съня
    sleep_quality["poor"] = fuzz.trimf(sleep_quality.universe, [0, 0, 40])
    sleep_quality["average"] = fuzz.trimf(sleep_quality.universe, [30, 50, 70])
    sleep_quality["good"] = fuzz.trimf(sleep_quality.universe, [60, 100, 100])

    # Fuzzy правила
    rule1 = ctrl.Rule(sleep_duration["long"] & noise["low"] & light["dark"], sleep_quality["good"])
    rule2 = ctrl.Rule(sleep_duration["normal"] & noise["low"] & light["dark"], sleep_quality["good"])
    rule3 = ctrl.Rule(noise["high"] & sleep_duration["short"], sleep_quality["poor"])
    rule4 = ctrl.Rule(light["bright"] & sleep_duration["short"], sleep_quality["poor"])
    rule5 = ctrl.Rule(sleep_duration["short"], sleep_quality["poor"])
    rule6 = ctrl.Rule(sleep_duration["normal"] & noise["medium"] & light["moderate"], sleep_quality["average"])
    rule7 = ctrl.Rule(sleep_duration["long"] & noise["medium"], sleep_quality["average"])
    rule8 = ctrl.Rule(sleep_duration["normal"] & noise["low"] & light["moderate"], sleep_quality["good"])
    rule9 = ctrl.Rule(sleep_duration["long"] & light["bright"], sleep_quality["average"])
    rule10 = ctrl.Rule(sleep_duration["short"] & noise["high"], sleep_quality["poor"])
    rule11 = ctrl.Rule(sleep_duration["long"] & noise["high"] & light["dark"], sleep_quality["average"])
    rule12 = ctrl.Rule(sleep_duration["long"] & noise["low"] & light["bright"], sleep_quality["average"])

    control_system = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5,
        rule6, rule7, rule8, rule9, rule10,
        rule11, rule12
    ])

    variables = {
        "sleep_duration": sleep_duration,
        "noise": noise,
        "light": light,
        "sleep_quality": sleep_quality
    }

    return variables, control_system


def quality_label(value):
    if value < 40:
        return "Лошо качество на съня"
    elif value < 70:
        return "Средно качество на съня"
    return "Добро качество на съня"


def run_simulation(control_system, sleep_duration_value, noise_value, light_value):
    simulation = ctrl.ControlSystemSimulation(control_system)
    simulation.input["sleep_duration"] = sleep_duration_value
    simulation.input["noise"] = noise_value
    simulation.input["light"] = light_value
    simulation.compute()

    result = simulation.output["sleep_quality"]
    label = quality_label(result)

    return result, label


def plot_membership_functions(variables, output_path):
    term_names_bg = {
        "short": "Кратък",
        "normal": "Нормален",
        "long": "Дълъг",
        "low": "Нисък",
        "medium": "Среден",
        "high": "Висок",
        "dark": "Тъмно",
        "moderate": "Умерено",
        "bright": "Ярко",
        "poor": "Лошо",
        "average": "Средно",
        "good": "Добро"
    }

    configs = [
        ("Продължителност на съня (часове)", variables["sleep_duration"]),
        ("Ниво на шум", variables["noise"]),
        ("Ниво на светлина", variables["light"]),
        ("Качество на съня", variables["sleep_quality"])
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 14))

    for ax, (title, variable) in zip(axes, configs):
        for term_name, term in variable.terms.items():
            ax.plot(
                variable.universe,
                term.mf,
                linewidth=2,
                label=term_names_bg.get(term_name, term_name)
            )

        ax.set_title(title)
        ax.set_xlabel("Стойност")
        ax.set_ylabel("Степен на принадлежност")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_result_example(variables, result, sample_inputs, output_path):
    term_names_bg = {
        "poor": "Лошо",
        "average": "Средно",
        "good": "Добро"
    }

    quality_var = variables["sleep_quality"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for term_name, term in quality_var.terms.items():
        ax.plot(
            quality_var.universe,
            term.mf,
            linewidth=2,
            label=term_names_bg.get(term_name, term_name)
        )

    ax.axvline(result, linestyle="--", linewidth=2, label=f"Резултат: {result:.2f}")

    info_text = (
        f"Примерни входни стойности:\n"
        f"Продължителност: {sample_inputs['sleep_duration']} ч.\n"
        f"Шум: {sample_inputs['noise']}\n"
        f"Светлина: {sample_inputs['light']}\n"
        f"Интерпретация: {quality_label(result)}"
    )

    ax.text(
        0.68, 0.35, info_text,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    ax.set_title("Примерен резултат за качеството на съня")
    ax.set_xlabel("Стойност на качеството на съня")
    ax.set_ylabel("Степен на принадлежност")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("screenshots", exist_ok=True)

    variables, control_system = build_fuzzy_system()

    # Генериране на изображение с membership функциите
    plot_membership_functions(
        variables,
        os.path.join("screenshots", "membership_functions.png")
    )

    test_cases = [
        {
            "name": "Тест 1",
            "sleep_duration": 8,
            "noise": 10,
            "light": 10,
            "expected": "Добро качество на съня"
        },
        {
            "name": "Тест 2",
            "sleep_duration": 6,
            "noise": 50,
            "light": 50,
            "expected": "Средно качество на съня"
        },
        {
            "name": "Тест 3",
            "sleep_duration": 4,
            "noise": 85,
            "light": 90,
            "expected": "Лошо качество на съня"
        },
        {
            "name": "Тест 4",
            "sleep_duration": 9,
            "noise": 70,
            "light": 20,
            "expected": "Средно качество на съня"
        },
        {
            "name": "Тест 5",
            "sleep_duration": 10,
            "noise": 15,
            "light": 70,
            "expected": "Средно качество на съня"
        }
    ]

    print("СИСТЕМА ЗА ОЦЕНКА НА КАЧЕСТВОТО НА СЪНЯ ЧРЕЗ РАЗМИТА ЛОГИКА")
    print("=" * 65)

    for case in test_cases:
        result, label = run_simulation(
            control_system,
            case["sleep_duration"],
            case["noise"],
            case["light"]
        )

        print(f"\n{case['name']}")
        print("-" * 30)
        print(f"Продължителност на съня: {case['sleep_duration']} часа")
        print(f"Ниво на шум: {case['noise']}")
        print(f"Ниво на светлина: {case['light']}")
        print(f"Очакван резултат: {case['expected']}")
        print(f"Получен резултат: {result:.2f}")
        print(f"Интерпретация: {label}")

    # Генериране на примерна визуализация на резултат
    example_case = test_cases[0]
    example_result, _ = run_simulation(
        control_system,
        example_case["sleep_duration"],
        example_case["noise"],
        example_case["light"]
    )

    plot_result_example(
        variables,
        example_result,
        example_case,
        os.path.join("screenshots", "result_example.png")
    )

    print("\nСъздадени файлове:")
    print("- screenshots/membership_functions.png")
    print("- screenshots/result_example.png")


if __name__ == "__main__":
    main()