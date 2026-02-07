"""
Test de job_matcher.py avec Approche 4
Lancer depuis la racine du projet : python -m tests.test_job_matcher_approach4
"""

from src.job_matcher import JobMatcher

# Initialiser
print("🔧 Initialisation du JobMatcher...")
matcher = JobMatcher()

# CV de test
cv_skills = ["Python", "Django", "PostgreSQL", "Docker", "Git", "Excel"]

# Offre de test
job = {
    "job_id": "test_001",
    "title": "Développeur Python/Django",
    "company": "Tech Startup",
    "location": "Paris",
    "description": "Nous recherchons un développeur Python/Django.",
    "requirements": [
        "Python (Django, Flask)",
        "PostgreSQL ou MySQL",
        "Docker",
        "Git"
    ],
    "nice_to_have": [
        "AWS",
        "Kubernetes"
    ]
}

print("\n" + "="*60)
print("🧪 TEST APPROCHE 4 : Skills Offre → CV")
print("="*60)

print(f"\n📄 CV Skills ({len(cv_skills)}) :")
for skill in cv_skills:
    print(f"  • {skill}")

print(f"\n💼 Job Requirements ({len(job['requirements'])}) :")
for req in job['requirements']:
    print(f"  • {req}")

# Calculer le matching
print("\n" + "="*60)
result = matcher.calculate_job_match_score(cv_skills, job)

print("\n" + "="*60)
print("📊 RÉSULTATS")
print("="*60)

print(f"\n🎯 Score Global : {result['score']:.1f}%")
print(f"   • Coverage : {result['skills_details']['coverage']:.1f}%")
print(f"   • Quality  : {result['skills_details']['quality']:.1f}%")
print(f"   • Couverts : {result['skills_details']['covered_count']}/{result['skills_details']['total_required']}")

print("\n🔧 Top Matches :")
for i, match in enumerate(result['skills_details']['top_matches'], 1):
    print(f"  {i}. {match['job_skill']:20} → {match['cv_skill']:20} ({match['similarity']:.1f}%)")

# Recalculer tous les matches (sans filtrage)
all_matches = matcher.calculate_skills_similarity(
    [matcher._normalize_skill(s) for s in cv_skills],
    job
)['matches']

print("\n🔧 Tous les skills de l'offre :")
for i, match in enumerate(all_matches, 1):
    cv_skill = match['cv_skill'] if match['cv_skill'] else "—"
    similarity = f"{match['similarity']:.1f}%"
    print(f"  {i}. {match['job_skill']:25} → {cv_skill:20} ({similarity})")


print("\n✅ Test terminé")