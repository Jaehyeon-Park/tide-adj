#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include <meep.hpp>
#include <meep/mympi.hpp>

#include <complex>
#include <cstdint>
#include <climits>
#include <vector>

#ifndef NPY_CSETREAL
#define NPY_CSETREAL(c, r) ((c)->real = (r))
#endif
#ifndef NPY_CSETIMAG
#define NPY_CSETIMAG(c, i) ((c)->imag = (i))
#endif

static std::complex<double> npy_to_complex(const npy_cdouble &value) {
    return std::complex<double>(npy_creal(value), npy_cimag(value));
}

static void set_npy_complex(npy_cdouble &target, const std::complex<double> &value) {
    NPY_CSETREAL(&target, value.real());
    NPY_CSETIMAG(&target, value.imag());
}

static void add_npy_complex(npy_cdouble &target, const std::complex<double> &value) {
    NPY_CSETREAL(&target, npy_creal(target) + value.real());
    NPY_CSETIMAG(&target, npy_cimag(target) + value.imag());
}

static int read_double_sequence(PyObject *obj, std::vector<double> &out, const char *name) {
    PyObject *seq = PySequence_Fast(obj, name);
    if (!seq) {
        return -1;
    }

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    out.clear();
    out.reserve(static_cast<size_t>(n));

    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        double value = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return -1;
        }
        out.push_back(value);
    }

    Py_DECREF(seq);
    return 0;
}

static PyObject *sample_component_grid(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);

    if (!PyArg_ParseTuple(args, "KOO|i", &fields_addr, &xs_obj, &ys_obj, &component_int)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (component_int < static_cast<int>(meep::Ex) ||
        component_int >= static_cast<int>(meep::NO_COMPONENT)) {
        PyErr_SetString(PyExc_ValueError, "component must be a Meep field component");
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    npy_intp dims[2] = {
        static_cast<npy_intp>(xs.size()),
        static_cast<npy_intp>(ys.size()),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    meep::fields *fields = reinterpret_cast<meep::fields *>(
        static_cast<uintptr_t>(fields_addr)
    );
    meep::component component = static_cast<meep::component>(component_int);
    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj))
    );

    size_t k = 0;
    try {
        for (double x : xs) {
            for (double y : ys) {
                std::complex<double> value = fields->get_field(component, meep::vec(x, y), true);
                set_npy_complex(data[k], value);
                ++k;
            }
        }
    } catch (const std::exception &e) {
        Py_DECREF(arr_obj);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(arr_obj);
        PyErr_SetString(PyExc_RuntimeError, "unknown error in Meep field sampling");
        return nullptr;
    }

    return arr_obj;
}

static std::complex<double> sample_component_local(
    meep::fields *fields,
    meep::component component,
    double x,
    double y
) {
    meep::ivec locs[8];
    double weights[8];
    fields->gv.interpolate(component, meep::vec(x, y), locs, weights);

    std::complex<double> value(0.0, 0.0);
    for (int p = 0; p < 8; ++p) {
        if (weights[p] == 0.0) {
            continue;
        }
        for (int chunk_idx = 0; chunk_idx < fields->num_chunks; ++chunk_idx) {
            meep::fields_chunk *chunk = fields->chunks[chunk_idx];
            if (!chunk || !chunk->is_mine() || !chunk->have_component(component)) {
                continue;
            }
            if (chunk->gv.owns(locs[p])) {
                value += weights[p] * chunk->get_field(component, locs[p]);
                break;
            }
        }
    }
    return value;
}

static constexpr const char *COMPONENT_GRID_PLAN_CAPSULE = "native_sampler.ComponentGridPlan";

struct SampleEntry {
    int chunk_idx;
    meep::ivec loc;
    double weight;
};

struct ComponentGridPlan {
    meep::fields *fields;
    meep::component component;
    size_t nx;
    size_t ny;
    std::vector<std::vector<SampleEntry>> points;
};

static void component_grid_plan_destructor(PyObject *capsule) {
    void *ptr = PyCapsule_GetPointer(capsule, COMPONENT_GRID_PLAN_CAPSULE);
    if (!ptr) {
        PyErr_Clear();
        return;
    }
    delete reinterpret_cast<ComponentGridPlan *>(ptr);
}

static ComponentGridPlan *get_component_grid_plan(PyObject *obj) {
    return reinterpret_cast<ComponentGridPlan *>(
        PyCapsule_GetPointer(obj, COMPONENT_GRID_PLAN_CAPSULE)
    );
}

static std::complex<double> sample_component_plan_local(
    const ComponentGridPlan *plan,
    size_t point_idx
) {
    std::complex<double> value(0.0, 0.0);
    for (const SampleEntry &entry : plan->points[point_idx]) {
        meep::fields_chunk *chunk = plan->fields->chunks[entry.chunk_idx];
        if (!chunk || !chunk->is_mine() || !chunk->have_component(plan->component)) {
            continue;
        }
        value += entry.weight * chunk->get_field(plan->component, entry.loc);
    }
    return value;
}

static PyObject *create_component_grid_plan(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);

    if (!PyArg_ParseTuple(args, "KOO|i", &fields_addr, &xs_obj, &ys_obj, &component_int)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (component_int < static_cast<int>(meep::Ex) ||
        component_int >= static_cast<int>(meep::NO_COMPONENT)) {
        PyErr_SetString(PyExc_ValueError, "component must be a Meep field component");
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    size_t total_size = xs.size() * ys.size();
    if (total_size > static_cast<size_t>(INT_MAX)) {
        PyErr_SetString(PyExc_OverflowError, "sample grid is too large for Meep MPI reduction");
        return nullptr;
    }

    meep::fields *fields = reinterpret_cast<meep::fields *>(
        static_cast<uintptr_t>(fields_addr)
    );
    meep::component component = static_cast<meep::component>(component_int);

    ComponentGridPlan *plan = new ComponentGridPlan;
    plan->fields = fields;
    plan->component = component;
    plan->nx = xs.size();
    plan->ny = ys.size();
    plan->points.resize(total_size);

    try {
        size_t k = 0;
        for (double x : xs) {
            for (double y : ys) {
                meep::ivec locs[8];
                double weights[8];
                fields->gv.interpolate(component, meep::vec(x, y), locs, weights);

                for (int p = 0; p < 8; ++p) {
                    if (weights[p] == 0.0) {
                        continue;
                    }
                    for (int chunk_idx = 0; chunk_idx < fields->num_chunks; ++chunk_idx) {
                        meep::fields_chunk *chunk = fields->chunks[chunk_idx];
                        if (!chunk || !chunk->is_mine() || !chunk->have_component(component)) {
                            continue;
                        }
                        if (chunk->gv.owns(locs[p])) {
                            plan->points[k].push_back({chunk_idx, locs[p], weights[p]});
                            break;
                        }
                    }
                }
                ++k;
            }
        }
    } catch (const std::exception &e) {
        delete plan;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        delete plan;
        PyErr_SetString(PyExc_RuntimeError, "unknown error while creating Meep sample plan");
        return nullptr;
    }

    PyObject *capsule = PyCapsule_New(plan, COMPONENT_GRID_PLAN_CAPSULE, component_grid_plan_destructor);
    if (!capsule) {
        delete plan;
        return nullptr;
    }
    return capsule;
}

static PyObject *sample_component_grid_plan_local_sum(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O", &plan_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    size_t total_size = plan->nx * plan->ny;
    std::vector<std::complex<double>> local(total_size, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> reduced(total_size, std::complex<double>(0.0, 0.0));

    try {
        for (size_t k = 0; k < total_size; ++k) {
            local[k] = sample_component_plan_local(plan, k);
        }

        meep::sum_to_all(
            local.data(),
            reduced.data(),
            static_cast<int>(total_size)
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown error in planned Meep field sampling");
        return nullptr;
    }

    npy_intp dims[2] = {
        static_cast<npy_intp>(plan->nx),
        static_cast<npy_intp>(plan->ny),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj))
    );
    for (size_t k = 0; k < total_size; ++k) {
        set_npy_complex(data[k], reduced[k]);
    }

    return arr_obj;
}

static PyObject *sample_component_grid_local_sum(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);

    if (!PyArg_ParseTuple(args, "KOO|i", &fields_addr, &xs_obj, &ys_obj, &component_int)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (component_int < static_cast<int>(meep::Ex) ||
        component_int >= static_cast<int>(meep::NO_COMPONENT)) {
        PyErr_SetString(PyExc_ValueError, "component must be a Meep field component");
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    size_t total_size = xs.size() * ys.size();
    if (total_size > static_cast<size_t>(INT_MAX)) {
        PyErr_SetString(PyExc_OverflowError, "sample grid is too large for Meep MPI reduction");
        return nullptr;
    }

    std::vector<std::complex<double>> local(total_size, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> reduced(total_size, std::complex<double>(0.0, 0.0));

    meep::fields *fields = reinterpret_cast<meep::fields *>(
        static_cast<uintptr_t>(fields_addr)
    );
    meep::component component = static_cast<meep::component>(component_int);

    try {
        size_t k = 0;
        for (double x : xs) {
            for (double y : ys) {
                local[k] = sample_component_local(fields, component, x, y);
                ++k;
            }
        }

        meep::sum_to_all(
            local.data(),
            reduced.data(),
            static_cast<int>(total_size)
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown error in local Meep field sampling");
        return nullptr;
    }

    npy_intp dims[2] = {
        static_cast<npy_intp>(xs.size()),
        static_cast<npy_intp>(ys.size()),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj))
    );
    for (size_t k = 0; k < total_size; ++k) {
        set_npy_complex(data[k], reduced[k]);
    }

    return arr_obj;
}

static PyObject *accumulate_component_product_local_sum(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);
    PyObject *multiplier_obj = nullptr;

    if (!PyArg_ParseTuple(args, "KOOiO", &fields_addr, &xs_obj, &ys_obj,
                          &component_int, &multiplier_obj)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (component_int < static_cast<int>(meep::Ex) ||
        component_int >= static_cast<int>(meep::NO_COMPONENT)) {
        PyErr_SetString(PyExc_ValueError, "component must be a Meep field component");
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    PyArrayObject *multiplier = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(multiplier_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY)
    );
    if (!multiplier) {
        return nullptr;
    }

    if (PyArray_NDIM(multiplier) != 2 ||
        PyArray_DIM(multiplier, 0) != static_cast<npy_intp>(xs.size()) ||
        PyArray_DIM(multiplier, 1) != static_cast<npy_intp>(ys.size())) {
        Py_DECREF(multiplier);
        PyErr_SetString(PyExc_ValueError, "multiplier must have shape (len(coords_x), len(coords_y))");
        return nullptr;
    }

    size_t total_size = xs.size() * ys.size();
    if (total_size > static_cast<size_t>(INT_MAX)) {
        Py_DECREF(multiplier);
        PyErr_SetString(PyExc_OverflowError, "sample grid is too large for Meep MPI reduction");
        return nullptr;
    }

    std::vector<std::complex<double>> local(total_size, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> reduced(total_size, std::complex<double>(0.0, 0.0));

    meep::fields *fields = reinterpret_cast<meep::fields *>(
        static_cast<uintptr_t>(fields_addr)
    );
    meep::component component = static_cast<meep::component>(component_int);
    npy_cdouble *mult_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(multiplier));

    try {
        size_t k = 0;
        for (double x : xs) {
            for (double y : ys) {
                std::complex<double> scale = npy_to_complex(mult_data[k]);
                local[k] = sample_component_local(fields, component, x, y) * scale;
                ++k;
            }
        }

        meep::sum_to_all(
            local.data(),
            reduced.data(),
            static_cast<int>(total_size)
        );
    } catch (const std::exception &e) {
        Py_DECREF(multiplier);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(multiplier);
        PyErr_SetString(PyExc_RuntimeError, "unknown error in local Meep product accumulation");
        return nullptr;
    }

    Py_DECREF(multiplier);

    npy_intp dims[2] = {
        static_cast<npy_intp>(xs.size()),
        static_cast<npy_intp>(ys.size()),
    };
    PyObject *arr_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!arr_obj) {
        return nullptr;
    }

    npy_cdouble *data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr_obj))
    );
    for (size_t k = 0; k < total_size; ++k) {
        set_npy_complex(data[k], reduced[k]);
    }

    return arr_obj;
}

static PyObject *accumulate_component_product_local_inplace(PyObject *, PyObject *args) {
    unsigned long long fields_addr = 0;
    PyObject *xs_obj = nullptr;
    PyObject *ys_obj = nullptr;
    int component_int = static_cast<int>(meep::Ez);
    PyObject *multiplier_obj = nullptr;
    PyObject *accumulator_obj = nullptr;

    if (!PyArg_ParseTuple(args, "KOOiOO", &fields_addr, &xs_obj, &ys_obj,
                          &component_int, &multiplier_obj, &accumulator_obj)) {
        return nullptr;
    }

    if (fields_addr == 0) {
        PyErr_SetString(PyExc_ValueError, "fields pointer address must be non-zero");
        return nullptr;
    }

    if (component_int < static_cast<int>(meep::Ex) ||
        component_int >= static_cast<int>(meep::NO_COMPONENT)) {
        PyErr_SetString(PyExc_ValueError, "component must be a Meep field component");
        return nullptr;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    if (read_double_sequence(xs_obj, xs, "coords_x must be a sequence") < 0 ||
        read_double_sequence(ys_obj, ys, "coords_y must be a sequence") < 0) {
        return nullptr;
    }

    PyArrayObject *multiplier = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(multiplier_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY)
    );
    if (!multiplier) {
        return nullptr;
    }

    PyArrayObject *accumulator = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(accumulator_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY)
    );
    if (!accumulator) {
        Py_DECREF(multiplier);
        return nullptr;
    }

    bool shape_ok = PyArray_NDIM(multiplier) == 2 &&
                    PyArray_NDIM(accumulator) == 2 &&
                    PyArray_DIM(multiplier, 0) == static_cast<npy_intp>(xs.size()) &&
                    PyArray_DIM(multiplier, 1) == static_cast<npy_intp>(ys.size()) &&
                    PyArray_DIM(accumulator, 0) == static_cast<npy_intp>(xs.size()) &&
                    PyArray_DIM(accumulator, 1) == static_cast<npy_intp>(ys.size());
    if (!shape_ok) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_ValueError, "multiplier and accumulator must have shape (len(coords_x), len(coords_y))");
        return nullptr;
    }

    meep::fields *fields = reinterpret_cast<meep::fields *>(
        static_cast<uintptr_t>(fields_addr)
    );
    meep::component component = static_cast<meep::component>(component_int);
    npy_cdouble *mult_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(multiplier));
    npy_cdouble *accum_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator));

    try {
        size_t k = 0;
        for (double x : xs) {
            for (double y : ys) {
                std::complex<double> scale = npy_to_complex(mult_data[k]);
                std::complex<double> value = sample_component_local(fields, component, x, y) * scale;
                add_npy_complex(accum_data[k], value);
                ++k;
            }
        }
    } catch (const std::exception &e) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, "unknown error in local Meep product accumulation");
        return nullptr;
    }

    Py_DECREF(multiplier);
    if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
        Py_DECREF(accumulator);
        return nullptr;
    }
    Py_DECREF(accumulator);

    Py_RETURN_NONE;
}

static PyObject *accumulate_component_product_plan_local_inplace(PyObject *, PyObject *args) {
    PyObject *plan_obj = nullptr;
    PyObject *multiplier_obj = nullptr;
    PyObject *accumulator_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OOO", &plan_obj, &multiplier_obj, &accumulator_obj)) {
        return nullptr;
    }

    ComponentGridPlan *plan = get_component_grid_plan(plan_obj);
    if (!plan) {
        return nullptr;
    }

    PyArrayObject *multiplier = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(multiplier_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY)
    );
    if (!multiplier) {
        return nullptr;
    }

    PyArrayObject *accumulator = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(accumulator_obj, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY)
    );
    if (!accumulator) {
        Py_DECREF(multiplier);
        return nullptr;
    }

    bool shape_ok = PyArray_NDIM(multiplier) == 2 &&
                    PyArray_NDIM(accumulator) == 2 &&
                    PyArray_DIM(multiplier, 0) == static_cast<npy_intp>(plan->nx) &&
                    PyArray_DIM(multiplier, 1) == static_cast<npy_intp>(plan->ny) &&
                    PyArray_DIM(accumulator, 0) == static_cast<npy_intp>(plan->nx) &&
                    PyArray_DIM(accumulator, 1) == static_cast<npy_intp>(plan->ny);
    if (!shape_ok) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_ValueError, "multiplier and accumulator must match the sample plan shape");
        return nullptr;
    }

    size_t total_size = plan->nx * plan->ny;
    npy_cdouble *mult_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(multiplier));
    npy_cdouble *accum_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(accumulator));

    try {
        for (size_t k = 0; k < total_size; ++k) {
            std::complex<double> scale = npy_to_complex(mult_data[k]);
            std::complex<double> value = sample_component_plan_local(plan, k) * scale;
            add_npy_complex(accum_data[k], value);
        }
    } catch (const std::exception &e) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(multiplier);
        PyArray_DiscardWritebackIfCopy(accumulator);
        Py_DECREF(accumulator);
        PyErr_SetString(PyExc_RuntimeError, "unknown error in planned local Meep product accumulation");
        return nullptr;
    }

    Py_DECREF(multiplier);
    if (PyArray_ResolveWritebackIfCopy(accumulator) < 0) {
        Py_DECREF(accumulator);
        return nullptr;
    }
    Py_DECREF(accumulator);

    Py_RETURN_NONE;
}

static PyObject *reduce_complex_grid_sum(PyObject *, PyObject *args) {
    PyObject *local_obj = nullptr;

    if (!PyArg_ParseTuple(args, "O", &local_obj)) {
        return nullptr;
    }

    PyArrayObject *local_arr = reinterpret_cast<PyArrayObject *>(
        PyArray_FROM_OTF(local_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY)
    );
    if (!local_arr) {
        return nullptr;
    }

    if (PyArray_NDIM(local_arr) != 2) {
        Py_DECREF(local_arr);
        PyErr_SetString(PyExc_ValueError, "local grid must be a 2D complex128 array");
        return nullptr;
    }

    npy_intp dims[2] = {
        PyArray_DIM(local_arr, 0),
        PyArray_DIM(local_arr, 1),
    };
    size_t total_size = static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]);
    if (total_size > static_cast<size_t>(INT_MAX)) {
        Py_DECREF(local_arr);
        PyErr_SetString(PyExc_OverflowError, "local grid is too large for Meep MPI reduction");
        return nullptr;
    }

    PyObject *reduced_obj = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    if (!reduced_obj) {
        Py_DECREF(local_arr);
        return nullptr;
    }

    npy_cdouble *local_data = reinterpret_cast<npy_cdouble *>(PyArray_DATA(local_arr));
    npy_cdouble *reduced_data = reinterpret_cast<npy_cdouble *>(
        PyArray_DATA(reinterpret_cast<PyArrayObject *>(reduced_obj))
    );

    try {
        meep::sum_to_all(
            reinterpret_cast<std::complex<double> *>(local_data),
            reinterpret_cast<std::complex<double> *>(reduced_data),
            static_cast<int>(total_size)
        );
    } catch (const std::exception &e) {
        Py_DECREF(local_arr);
        Py_DECREF(reduced_obj);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        Py_DECREF(local_arr);
        Py_DECREF(reduced_obj);
        PyErr_SetString(PyExc_RuntimeError, "unknown error in Meep grid reduction");
        return nullptr;
    }

    Py_DECREF(local_arr);
    return reduced_obj;
}

static PyMethodDef FastMeepSampleMethods[] = {
    {
        "create_component_grid_plan",
        create_component_grid_plan,
        METH_VARARGS,
        "Precompute rank-local interpolation support for a Meep field component over xs x ys.",
    },
    {
        "sample_component_grid_plan_local_sum",
        sample_component_grid_plan_local_sum,
        METH_VARARGS,
        "Sample a Meep field component using a precomputed rank-local plan and one MPI all-rank sum.",
    },
    {
        "sample_component_grid",
        sample_component_grid,
        METH_VARARGS,
        "Sample a Meep field component over xs x ys using meep::fields::get_field.",
    },
    {
        "sample_component_grid_local_sum",
        sample_component_grid_local_sum,
        METH_VARARGS,
        "Sample a Meep field component using rank-local chunks and one MPI all-rank sum.",
    },
    {
        "accumulate_component_product_local_sum",
        accumulate_component_product_local_sum,
        METH_VARARGS,
        "Accumulate field-component times a complex grid using rank-local chunks and one MPI all-rank sum.",
    },
    {
        "accumulate_component_product_local_inplace",
        accumulate_component_product_local_inplace,
        METH_VARARGS,
        "Accumulate field-component times a complex grid into a rank-local accumulator without MPI reduction.",
    },
    {
        "accumulate_component_product_plan_local_inplace",
        accumulate_component_product_plan_local_inplace,
        METH_VARARGS,
        "Accumulate field-component times a complex grid with a precomputed rank-local plan and no MPI reduction.",
    },
    {
        "reduce_complex_grid_sum",
        reduce_complex_grid_sum,
        METH_VARARGS,
        "Sum a complex grid across all MPI ranks and return the reduced grid.",
    },
    {
        "sample_ez_grid",
        sample_component_grid,
        METH_VARARGS,
        "Compatibility alias for sample_component_grid.",
    },
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef native_sampler_module = {
    PyModuleDef_HEAD_INIT,
    "native_sampler",
    "Native batch sampler for Meep field points.",
    -1,
    FastMeepSampleMethods,
};

PyMODINIT_FUNC PyInit_native_sampler(void) {
    import_array();
    return PyModule_Create(&native_sampler_module);
}
