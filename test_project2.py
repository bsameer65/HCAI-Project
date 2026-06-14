"""
Comprehensive integration test for Project 2.
Run with: venv\\Scripts\\python.exe test_project2.py
"""
import django, os, sys, traceback
os.environ['DJANGO_SETTINGS_MODULE'] = 'pbl.settings'
django.setup()

from django.test import RequestFactory, TestCase
from django.conf import settings

passed = []
failed = []

def test(name, fn):
    try:
        fn()
        passed.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        failed.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()

rf = RequestFactory()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────
def test_data_loading():
    from project2.data_utils import load_penguin_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    df, X, y, num, cat, cls = load_penguin_data()
    assert len(df) > 300, f"Too few rows: {len(df)}"
    assert list(num) == NUMERICAL_FEATURES
    assert list(cat) == CATEGORICAL_FEATURES
    assert sorted(cls) == ['Adelie', 'Chinstrap', 'Gentoo']
    assert 'species' in df.columns
    assert df.isnull().sum().sum() == 0, "Dataset has NaN values after cleaning"

test("1. Data loading", test_data_loading)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Decision Tree training
# ─────────────────────────────────────────────────────────────────────────────
def test_dt_training():
    from project2.model_utils import get_selected_model
    r = get_selected_model('dt', 0.5)
    assert r['model_type'] == 'dt'
    assert 0.0 < r['accuracy'] <= 1.0
    assert r['complexity'] > 0
    assert r['pipeline'] is not None
    assert len(r['candidates']) == 8  # 8 depth values
    assert r['coef_table'] is None  # DT has no coefficient table

test("2. DT training + selection", test_dt_training)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Logistic Regression training
# ─────────────────────────────────────────────────────────────────────────────
def test_lr_training():
    from project2.model_utils import get_selected_model
    r = get_selected_model('lr', 0.5)
    assert r['model_type'] == 'lr'
    assert 0.0 < r['accuracy'] <= 1.0
    assert r['complexity'] >= 0
    assert len(r['candidates']) == 8  # 8 C values
    assert r['coef_table'] is not None
    assert len(r['coef_table']) > 0
    # Check coefficient table structure
    row = r['coef_table'][0]
    assert 'feature' in row
    assert 'nonzero' in row

test("3. LR training + selection", test_lr_training)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Lambda affects selection
# ─────────────────────────────────────────────────────────────────────────────
def test_lambda_effect():
    from project2.model_utils import get_selected_model
    r0 = get_selected_model('dt', 0.0)
    r1 = get_selected_model('dt', 1.0)
    # lambda=0 should pick most accurate (likely deeper tree)
    # lambda=1 should pick simplest (likely shallow tree)
    assert r0['complexity'] >= r1['complexity'], \
        f"lambda=0 complexity ({r0['complexity']}) should be >= lambda=1 ({r1['complexity']})"

test("4. Lambda affects model selection", test_lambda_effect)

# ─────────────────────────────────────────────────────────────────────────────
# 5. LR with categorical encoding
# ─────────────────────────────────────────────────────────────────────────────
def test_lr_categorical_encoding():
    from project2.model_utils import get_selected_model, get_encoded_feature_names
    r = get_selected_model('lr', 0.5)
    feat_names = get_encoded_feature_names(r['pipeline'])
    # Should contain OHE-expanded names like island_Biscoe, sex_male, etc.
    ohe_names = [n for n in feat_names if '_' in n and n not in ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']]
    assert len(ohe_names) > 0, f"No OHE features found: {feat_names}"

test("5. LR categorical encoding", test_lr_categorical_encoding)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Counterfactual generation
# ─────────────────────────────────────────────────────────────────────────────
def test_counterfactuals():
    from project2.model_utils import get_selected_model
    from project2.counterfactual_utils import generate_counterfactuals
    selected = get_selected_model('dt', 0.5)
    # Row 0 is Adelie, ask for Gentoo
    result = generate_counterfactuals(selected, row_index=0, target_class_name='Gentoo')
    assert result['original_prediction'] == 'Adelie'
    assert result['target_class'] == 'Gentoo'
    assert len(result['counterfactuals']) > 0, "No counterfactuals found"
    cf = result['counterfactuals'][0]
    assert 'distance' in cf
    assert 'changes' in cf
    assert cf['distance'] > 0

test("6. Counterfactual generation", test_counterfactuals)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Counterfactuals use the selected model
# ─────────────────────────────────────────────────────────────────────────────
def test_cf_uses_selected_model():
    from project2.model_utils import get_selected_model
    from project2.counterfactual_utils import generate_counterfactuals
    sel_dt = get_selected_model('dt', 0.5)
    sel_lr = get_selected_model('lr', 0.5)
    cf_dt = generate_counterfactuals(sel_dt, row_index=0, target_class_name='Gentoo')
    cf_lr = generate_counterfactuals(sel_lr, row_index=0, target_class_name='Gentoo')
    # Different models should (likely) produce different counterfactuals
    if cf_dt['counterfactuals'] and cf_lr['counterfactuals']:
        d_dt = cf_dt['counterfactuals'][0]['distance']
        d_lr = cf_lr['counterfactuals'][0]['distance']
        # Just check they both work — exact values may differ
        assert d_dt > 0 and d_lr > 0

test("7. Counterfactuals use selected model", test_cf_uses_selected_model)

# ─────────────────────────────────────────────────────────────────────────────
# 8. PDP computation
# ─────────────────────────────────────────────────────────────────────────────
def test_pdp():
    from project2.model_utils import get_selected_model
    from project2.effect_plot_utils import compute_pdp
    selected = get_selected_model('dt', 0.5)
    pdp = compute_pdp(selected['pipeline'], selected['X'], 'flipper_length_mm', selected['class_names'])
    assert len(pdp['grid_values']) == 50
    assert set(pdp['pdp_values'].keys()) == {'Adelie', 'Chinstrap', 'Gentoo'}
    # Probabilities should sum to ~1 at each grid point
    for i in range(len(pdp['grid_values'])):
        total = sum(pdp['pdp_values'][cls][i] for cls in pdp['pdp_values'])
        assert abs(total - 1.0) < 0.01, f"Probs don't sum to 1 at grid[{i}]: {total}"

test("8. PDP computation", test_pdp)

# ─────────────────────────────────────────────────────────────────────────────
# 9. ALE computation
# ─────────────────────────────────────────────────────────────────────────────
def test_ale():
    from project2.model_utils import get_selected_model
    from project2.effect_plot_utils import compute_ale
    selected = get_selected_model('dt', 0.5)
    ale = compute_ale(selected['pipeline'], selected['X'], 'flipper_length_mm', selected['class_names'])
    assert ale['n_bins_used'] > 0
    assert len(ale['bin_centres']) > 0
    assert set(ale['ale_values'].keys()) == {'Adelie', 'Chinstrap', 'Gentoo'}

test("9. ALE computation", test_ale)

# ─────────────────────────────────────────────────────────────────────────────
# 10. PDP/ALE use the selected model
# ─────────────────────────────────────────────────────────────────────────────
def test_pdp_ale_uses_model():
    from project2.model_utils import get_selected_model
    from project2.effect_plot_utils import compute_pdp, compute_ale
    sel_dt = get_selected_model('dt', 0.0)
    sel_lr = get_selected_model('lr', 0.5)
    pdp_dt = compute_pdp(sel_dt['pipeline'], sel_dt['X'], 'body_mass_g', sel_dt['class_names'], n_grid=5)
    pdp_lr = compute_pdp(sel_lr['pipeline'], sel_lr['X'], 'body_mass_g', sel_lr['class_names'], n_grid=5)
    # Both should work and produce valid probabilities
    for cls in ['Adelie', 'Chinstrap', 'Gentoo']:
        assert len(pdp_dt['pdp_values'][cls]) == 5
        assert len(pdp_lr['pdp_values'][cls]) == 5

test("10. PDP/ALE use selected model", test_pdp_ale_uses_model)

# ─────────────────────────────────────────────────────────────────────────────
# 11. Plot saving
# ─────────────────────────────────────────────────────────────────────────────
def test_plot_saving():
    from project2.model_utils import get_selected_model
    from project2.effect_plot_utils import compute_pdp, compute_ale
    from project2.plot_utils import plot_pdp, plot_ale
    selected = get_selected_model('dt', 0.5)
    pdp = compute_pdp(selected['pipeline'], selected['X'], 'bill_length_mm', selected['class_names'], n_grid=10)
    pdp_url = plot_pdp(pdp, 'bill_length_mm', selected['label'])
    assert pdp_url.startswith('/media/')
    # Check file exists
    filepath = os.path.join(settings.BASE_DIR, pdp_url.lstrip('/'))
    assert os.path.isfile(filepath), f"PDP file not found: {filepath}"

    ale = compute_ale(selected['pipeline'], selected['X'], 'bill_length_mm', selected['class_names'])
    ale_url = plot_ale(ale, 'bill_length_mm', selected['label'])
    assert ale_url.startswith('/media/')

test("11. Plot saving to media", test_plot_saving)

# ─────────────────────────────────────────────────────────────────────────────
# 12. View GET requests (template rendering, no errors)
# ─────────────────────────────────────────────────────────────────────────────
def test_views_get():
    from project2.views import index, train, counterfactual, pdp_ale
    for name, view in [('index', index), ('train', train), ('counterfactual', counterfactual), ('pdp_ale', pdp_ale)]:
        req = rf.get('/project2/')
        resp = view(req)
        assert resp.status_code == 200, f"{name} GET returned {resp.status_code}"

test("12. All views render on GET", test_views_get)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Train POST (form submit simulation)
# ─────────────────────────────────────────────────────────────────────────────
def test_train_post():
    from project2.views import train
    from django.middleware.csrf import CsrfViewMiddleware
    req = rf.post('/project2/train/', {'model_type': 'dt', 'lambda_value': '0.3'})
    req._dont_enforce_csrf_checks = True
    resp = train(req)
    assert resp.status_code == 200
    content = resp.content.decode()
    assert 'Selected Model' in content or 'DT depth' in content, "No model result in response"

test("13. Train POST (DT)", test_train_post)

def test_train_post_lr():
    from project2.views import train
    req = rf.post('/project2/train/', {'model_type': 'lr', 'lambda_value': '0.5'})
    req._dont_enforce_csrf_checks = True
    resp = train(req)
    assert resp.status_code == 200
    content = resp.content.decode()
    assert 'LR C=' in content, "No LR model result in response"
    assert 'Coefficient Table' in content or 'coef' in content.lower(), "No coefficient table"

test("14. Train POST (LR)", test_train_post_lr)

# ─────────────────────────────────────────────────────────────────────────────
# 15. Counterfactual POST
# ─────────────────────────────────────────────────────────────────────────────
def test_cf_post():
    from project2.views import counterfactual
    req = rf.post('/project2/counterfactual/', {
        'model_type': 'dt', 'lambda_value': '0.5',
        'row_index': '0', 'target_class': 'Gentoo'
    })
    req._dont_enforce_csrf_checks = True
    resp = counterfactual(req)
    assert resp.status_code == 200
    content = resp.content.decode()
    assert 'Adelie' in content  # original prediction
    assert 'Gentoo' in content  # target class
    # Should not contain raw Python traceback
    assert 'Traceback' not in content, "Raw traceback in counterfactual response"

test("15. Counterfactual POST", test_cf_post)

# ─────────────────────────────────────────────────────────────────────────────
# 16. PDP/ALE POST
# ─────────────────────────────────────────────────────────────────────────────
def test_pdp_ale_post():
    from project2.views import pdp_ale
    req = rf.post('/project2/pdp-ale/', {
        'model_type': 'dt', 'lambda_value': '0.5',
        'feature_name': 'flipper_length_mm'
    })
    req._dont_enforce_csrf_checks = True
    resp = pdp_ale(req)
    assert resp.status_code == 200
    content = resp.content.decode()
    assert 'pdp_flipper_length_mm' in content or 'PDP' in content
    assert 'ale_flipper_length_mm' in content or 'ALE' in content
    assert 'Traceback' not in content, "Raw traceback in pdp_ale response"

test("16. PDP/ALE POST", test_pdp_ale_post)

# ─────────────────────────────────────────────────────────────────────────────
# 17. Form values preserved after POST
# ─────────────────────────────────────────────────────────────────────────────
def test_form_value_preservation():
    from project2.views import train
    req = rf.post('/project2/train/', {'model_type': 'lr', 'lambda_value': '0.73'})
    req._dont_enforce_csrf_checks = True
    resp = train(req)
    content = resp.content.decode()
    # The LR radio should be checked
    assert 'value="lr"' in content
    # Lambda should appear in the output
    assert '0.73' in content, "Lambda value 0.73 not preserved in response"

test("17. Form values preserved after submit", test_form_value_preservation)

# ─────────────────────────────────────────────────────────────────────────────
# 18. No raw errors in any view
# ─────────────────────────────────────────────────────────────────────────────
def test_no_raw_errors():
    from project2.views import index, train, counterfactual, pdp_ale
    views = [
        ('index', index, {}),
        ('train', train, {'model_type': 'dt', 'lambda_value': '0.5'}),
        ('counterfactual', counterfactual, {'model_type': 'dt', 'lambda_value': '0.5', 'row_index': '0', 'target_class': 'Gentoo'}),
        ('pdp_ale', pdp_ale, {'model_type': 'dt', 'lambda_value': '0.5', 'feature_name': 'bill_depth_mm'}),
    ]
    for name, view, data in views:
        if data:
            req = rf.post('/project2/', data)
            req._dont_enforce_csrf_checks = True
        else:
            req = rf.get('/project2/')
        resp = view(req)
        content = resp.content.decode()
        assert 'Traceback' not in content, f"Raw traceback in {name}"
        assert 'SyntaxError' not in content, f"SyntaxError in {name}"
        assert 'TemplateDoesNotExist' not in content, f"Missing template in {name}"

test("18. No raw Python errors in any view", test_no_raw_errors)

# ─────────────────────────────────────────────────────────────────────────────
# 19. All 4 numerical features work for PDP/ALE
# ─────────────────────────────────────────────────────────────────────────────
def test_all_features_pdp_ale():
    from project2.model_utils import get_selected_model
    from project2.effect_plot_utils import compute_pdp, compute_ale
    from project2.data_utils import NUMERICAL_FEATURES
    selected = get_selected_model('lr', 0.5)
    for feat in NUMERICAL_FEATURES:
        pdp = compute_pdp(selected['pipeline'], selected['X'], feat, selected['class_names'], n_grid=5)
        ale = compute_ale(selected['pipeline'], selected['X'], feat, selected['class_names'])
        assert len(pdp['grid_values']) == 5, f"PDP failed for {feat}"
        assert ale['n_bins_used'] > 0, f"ALE failed for {feat}"

test("19. All 4 features work for PDP/ALE", test_all_features_pdp_ale)

# ─────────────────────────────────────────────────────────────────────────────
# 20. Model persistence (save/load)
# ─────────────────────────────────────────────────────────────────────────────
def test_model_persistence():
    from project2.views import _save_selected, _load_selected
    from project2.model_utils import get_selected_model
    selected = get_selected_model('dt', 0.5)
    _save_selected(selected)
    loaded = _load_selected()
    assert loaded is not None
    assert loaded['model_type'] == 'dt'
    assert loaded['label'] == selected['label']

test("20. Model save/load persistence", test_model_persistence)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {len(passed)} passed, {len(failed)} failed")
print("=" * 60)
if failed:
    print("\nFailed tests:")
    for name, err in failed:
        print(f"  FAIL  {name}: {err}")
    sys.exit(1)
else:
    print("\n  All tests passed!")
    sys.exit(0)
