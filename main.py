import pandas as pd
import numpy as np

# Student Performance Analytics System

def load_data(filename):
    """Load student records from a CSV file."""
    try:
        data = pd.read_csv(filename)
        print("\nStudent dataset loaded successfully!")
        return data
    except FileNotFoundError:
        print(f"Error: '{filename}' was not found.")
        print("Place students.csv in the same folder as main.py.")
        return None


def calculate_performance(data):
    """Calculate totals, averages, grades, and pass/fail status."""
    subjects = [
        "Mathematics",
        "Physics",
        "Chemistry",
        "English",
        "Computer_Science",
    ]

    # NumPy calculates total and average marks for each student.
    marks = data[subjects].to_numpy(dtype=float)
    data["Total_Marks"] = np.sum(marks, axis=1)
    data["Average_Marks"] = np.round(np.mean(marks, axis=1), 2)

    # Example grading rule: A >= 90, B >= 80, C >= 70,
    # D >= 60, E >= 50, otherwise F.
    def get_grade(average):
        if average >= 90:
            return "A"
        elif average >= 80:
            return "B"
        elif average >= 70:
            return "C"
        elif average >= 60:
            return "D"
        elif average >= 50:
            return "E"
        else:
            return "F"

    data["Grade"] = data["Average_Marks"].apply(get_grade)

    # Example rule: pass if every subject mark is at least 40.
    data["Result"] = np.where((data[subjects] >= 40).all(axis=1), "Pass", "Fail")
    return data, subjects


def show_summary(data, subjects):
    """Display class and subject summaries."""
    print("\n========== STUDENT PERFORMANCE ANALYTICS ==========")
    print(f"Number of students: {len(data)}")
    print(f"Class average: {data['Average_Marks'].mean():.2f}")

    top_student = data.loc[data["Average_Marks"].idxmax()]
    low_student = data.loc[data["Average_Marks"].idxmin()]
    print(f"\nTop student: {top_student['Name']} "
          f"(average: {top_student['Average_Marks']:.2f})")
    print(f"Lowest average: {low_student['Name']} "
          f"(average: {low_student['Average_Marks']:.2f})")

    print("\nSubject-wise averages:")
    subject_averages = data[subjects].mean().round(2)
    print(subject_averages.to_string())

    print("\nResult count:")
    print(data["Result"].value_counts().to_string())

    print("\nStudent results:")
    columns = ["Student_ID", "Name", "Department", "Total_Marks",
               "Average_Marks", "Grade", "Result", "Attendance_Percent"]
    print(data[columns].to_string(index=False))


def advanced_analysis(data, subjects):

    # 1. Display the top 5 students
    print("\n===== TOP 5 STUDENTS =====")

    top_five = data.sort_values(
        "Average_Marks", ascending=False
    ).head(5)

    print(top_five[
        ["Name", "Department", "Average_Marks", "Grade"]
    ].to_string(index=False))

    # 2. Subject-wise average
    print("\n===== SUBJECT ANALYSIS =====")

    subject_avg = data[subjects].mean().round(2)
    print(subject_avg.sort_values(ascending=False))

    print("Best subject:", subject_avg.idxmax())
    print("Lowest subject:", subject_avg.idxmin())

    # 3. Department comparison
    print("\n===== DEPARTMENT ANALYSIS =====")

    department_avg = data.groupby(
        "Department"
    )["Average_Marks"].mean().round(2)

    print(department_avg.sort_values(ascending=False))

    # 4. Grade distribution
    print("\n===== GRADE DISTRIBUTION =====")

    print(data["Grade"].value_counts().sort_index())

    # 5. Students with low attendance
    print("\n===== LOW ATTENDANCE =====")

    low_attendance = data[
        data["Attendance_Percent"] < 75
    ]

    if low_attendance.empty:
        print("No students have attendance below 75%.")
    else:
        print(low_attendance[
            ["Name", "Department", "Attendance_Percent"]
        ].to_string(index=False))


def main():
    data = load_data("students.csv")
    if data is None:
        return

    required_columns = [
        "Student_ID", "Name", "Department", "Mathematics", "Physics",
        "Chemistry", "English", "Computer_Science", "Attendance_Percent"
    ]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        print("Error: CSV is missing these columns:", ", ".join(missing))
        return

    data, subjects = calculate_performance(data)
    show_summary(data, subjects)
    advanced_analysis(data, subjects)

    # Save a new file containing the calculated results.
    data.to_csv("student_results.csv", index=False)
    print("\nDetailed results saved to 'student_results.csv'.")


if __name__ == "__main__":
    main()
