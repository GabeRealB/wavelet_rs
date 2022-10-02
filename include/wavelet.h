#ifndef WAVELET_H
#define WAVELET_H

///////////////// Compatiblity checks /////////////////

/// Check that the library was compiled with the required features.
/// Features:
///         - WAVELET_RS_FEAT_FFI ("ffi"): export ffi api.
///         - WAVELET_RS_FEAT_FFI_VEC ("ffi_vec"): export vector encoders and decoders.
///         - WAVELET_RS_FEAT_FFI_MAT ("ffi_mat"): export matrix encoders and decoders.
///         - WAVELET_RS_FEAT_FFI_MEATADATA_ARR ("ffi_metadata_arr"): export reading/writing of arrays from/to the metadata.
///         - WAVELET_RS_FEAT_FFI_MEATADATA_SLICE ("ffi_metadata_arr"): export reading/writing of slices from/to the metadata.
///
/// Import options:
///         - WAVELET_RS_IMPORT_ALL: enable all imports.
///         - WAVELET_RS_IMPORT_VEC: import vector encoder and decoder declarations.
///         - WAVELET_RS_IMPORT_MAT: import matrix encoder and decoder declarations.
///         - WAVELET_RS_IMPORT_MEATADATA_ARR: import array metadata declarations.
///         - WAVELET_RS_IMPORT_MEATADATA_SLICE: import matrix metadata declarations.

#define WAVELET_RS_FEAT_FFI

#ifndef WAVELET_RS_FEAT_FFI
#error "wavelet-rs must be compiled with the 'ffi' feature"
#endif // !WAVELET_FEAT_FFI

#ifdef WAVELET_RS_IMPORT_ALL
#define WAVELET_RS_IMPORT_VEC
#define WAVELET_RS_IMPORT_MAT
#define WAVELET_RS_IMPORT_MEATADATA_ARR
#define WAVELET_RS_IMPORT_MEATADATA_SLICE
#endif // WAVELET_RS_IMPORT_ALL

#if defined WAVELET_RS_IMPORT_VEC && !defined WAVELET_RS_FEAT_FFI_VEC
#error "wavelet-rs was not compiled with the required 'ffi_vec' feature (required because of 'WAVELET_RS_IMPORT_VEC')"
#endif // WAVELET_RS_IMPORT_VEC && !WAVELET_RS_FEAT_FFI_VEC

#if defined WAVELET_RS_IMPORT_MAT && !defined WAVELET_RS_FEAT_FFI_MAT
#error "wavelet-rs was not compiled with the required 'ffi_mat' feature (required because of 'WAVELET_RS_IMPORT_MAT')"
#endif // WAVELET_RS_IMPORT_MAT && !WAVELET_RS_FEAT_FFI_MAT

#if defined WAVELET_RS_IMPORT_MEATADATA_ARR && !defined WAVELET_RS_FEAT_FFI_MEATADATA_ARR
#error "wavelet-rs was not compiled with the required 'ffi_metadata_arr' feature (required because of 'WAVELET_RS_IMPORT_MEATADATA_ARR')"
#endif // WAVELET_RS_IMPORT_MEATADATA_ARR && !WAVELET_RS_FEAT_FFI_MEATADATA_ARR

#if defined WAVELET_RS_IMPORT_MEATADATA_SLICE && !defined WAVELET_RS_FEAT_FFI_MEATADATA_SLICE
#error "wavelet-rs was not compiled with the required 'ffi_metadata_arr' feature (required because of 'WAVELET_RS_IMPORT_MEATADATA_SLICE')"
#endif // WAVELET_RS_IMPORT_MEATADATA_SLICE && !WAVELET_RS_FEAT_FFI_MEATADATA_SLICE

///////////////// FFI API mapping /////////////////

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(sizeof(float) * CHAR_BIT == 32, "float must have a width of 32 bits");
static_assert(sizeof(double) * CHAR_BIT == 64, "double must have a width of 64 bits");
static_assert(std::numeric_limits<float>::is_iec559, "float must be a IEEE 754 floating point type");
static_assert(std::numeric_limits<double>::is_iec559, "double must be a IEEE 754 floating point type");

namespace wavelet {

/// Definition of a noninclusive range.
///
/// @tparam T range element type.
template <typename T>
struct range {
    T start;
    T end;
};

/// Constant view over a borrowed contiguous memory region.
///
/// @tparam T Element type.
template <typename T>
class slice {
    // Invariant: m_ptr is invalid or m_len > 0.
    const T* m_ptr;
    std::size_t m_len;

public:
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef const value_type& const_reference;
    typedef const value_type* const_pointer;
    typedef const value_type* const_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    /// Constructs a new empty slice.
    explicit slice() noexcept
        : m_ptr { reinterpret_cast<const T*>(alignof(value_type)) }
        , m_len { 0 }
    {
    }

    /// Constructs a new slice containing one element.
    ///
    /// @param value value the slice points to.
    explicit slice(const T& value) noexcept
        : m_ptr { &value }
        , m_len { 1 }
    {
    }

    /// Constructs a new slice from a pointer and a length.
    /// The pointer must be properly aligned and valid in the range [ptr; ptr+size].
    ///
    /// @param ptr start of the array.
    /// @param size length of the array.
    explicit slice(const T* ptr, size_type size) noexcept
        : m_ptr { ptr }
        , m_len { size }
    {
    }

    /// @brief Constructs a new slice pointing to an array.
    ///
    /// @tparam N size of the array.
    /// @param array array the slice points to.
    template <std::size_t N>
    explicit slice(const std::array<T, N>& array) noexcept
        : m_ptr { array.data() }
        , m_len { array.size() }
    {
    }

    slice(const slice& other) = default;
    slice(slice&& other) noexcept = default;

    slice& operator=(const slice& other) = default;
    slice& operator=(slice&& other) = default;

    /// Checks that the elements of the two slices are equal.
    /// Is equivalent to calling std::equal with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the two slices are equal, `false` otherwise.
    template <typename U>
    bool operator==(const slice<U>& other) const
    {
        return std::equal(this->begin(), this->end(), other.begin());
    }

    /// Checks that the elements of the two slices are equal.
    /// Is equivalent to negating the call to std::equal with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the two slices are not equal, `false` otherwise.
    template <typename U>
    bool operator!=(const slice<U>& other) const
    {
        return !(*this == other);
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the contents of this are lexicographically less than
    /// the contents of other, `false` otherwise.
    template <typename U>
    bool operator<(const slice<U>& other) const
    {
        return std::lexicographical_compare(
            this->begin(), this->end(),
            other.begin(), other.end());
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the contents of this are lexicographically less than
    /// or equal to the contents of other, `false` otherwise.
    template <typename U>
    bool operator<=(const slice<U>& other) const
    {
        return std::lexicographical_compare(
            this->begin(), this->end(),
            other.begin(), other.end(),
            [](const T& lhs, const T& rhs) { return lhs <= rhs; });
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the contents of this are lexicographically greater than
    /// the contents of other, `false` otherwise.
    template <typename U>
    bool operator>(const slice<U>& other) const
    {
        return std::lexicographical_compare(
            this->begin(), this->end(),
            other.begin(), other.end(),
            [](const T& lhs, const T& rhs) { return lhs > rhs; });
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the contents of this are lexicographically greater than
    /// or equal to the contents of other, `false` otherwise.
    template <typename U>
    bool operator>=(const slice<U>& other) const
    {
        return std::lexicographical_compare(
            this->begin(), this->end(),
            other.begin(), other.end(),
            [](const T& lhs, const T& rhs) { return lhs >= rhs; });
    }

    /// Returns a reference to the element at specified location pos, with bounds checking.
    /// If pos is not within the range of the container, an exception of type std::out_of_range
    /// is thrown.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    const_reference at(size_type pos) const
    {
        if (pos >= this->m_len)
            throw std::out_of_range("slice access out of range");
        return this[pos];
    }

    /// Returns a reference to the element at specified location pos.
    /// No bounds checking is performed.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    const_reference operator[](size_type pos) const
    {
        return this->m_ptr[pos];
    }

    /// Returns a reference to the first element in the container.
    /// @return Calling front on an empty container is undefined.
    const_reference front() const
    {
        return *this->begin();
    }

    /// Returns a reference to the last element in the container.
    /// @return Calling back on an empty container is undefined.
    const_reference back() const
    {
        return *std::prev(this->end());
    }

    /// Returns pointer to the underlying array serving as element storage.
    /// @return Pointer to the underlying element storage.
    const T* data() const noexcept
    {
        return this->m_ptr;
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    const_iterator begin() const noexcept
    {
        return const_iterator { this->m_ptr };
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    const_iterator cbegin() const noexcept
    {
        return const_iterator { this->m_ptr };
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    const_iterator end() const noexcept
    {
        return const_iterator { this->m_ptr + this->m_len };
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    const_iterator cend() const noexcept
    {
        return const_iterator { this->m_ptr + this->m_len };
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator { this->end() };
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator { this->cend() };
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator { this->begin() };
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator { this->cbegin() };
    }

    /// Checks if the container has no elements, i.e. whether `begin() == end()`.
    ///
    /// @return `true` if the container is empty, `false` otherwise.
    [[nodiscard]] bool empty() const noexcept
    {
        return this->begin() == this->end();
    }

    /// Returns the number of elements in the container, i.e. `std::distance(begin(), end())`.
    ///
    /// @return The number of elements in the container.
    [[nodiscard]] size_type size() const noexcept
    {
        return std::distance(this->begin(), this->end());
    }
};

/// Owned view over a sontiguous memory region.
///
/// @tparam T Element type.
template <typename T>
class owned_slice {
    // Invariant: m_ptr is invalid or m_len > 0.
    T* m_ptr;
    std::size_t m_len;

public:
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type* iterator;
    typedef const value_type* const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    // Delete constructors/destructors which (de)allocate, as
    // we have no general way of implementing it.

    /// Constructs a new slice by copying all elements of the provided slice.
    ///
    /// @param value slice to be copied into the new slice.
    explicit owned_slice(slice<T> s) = delete;
    ~owned_slice() = delete;

    /// Constructs a new empty slice.
    explicit owned_slice()
        : owned_slice { slice<T> {} }
    {
    }

    /// Constructs a new slice containing only one value.
    ///
    /// @param value Value contained.
    explicit owned_slice(const T& value)
        : owned_slice { slice<T> { value } }
    {
    }

    /// Constructs a new slice from a pointer and a length.
    /// The pointer must be properly aligned and valid in the range [ptr; ptr+size].
    ///
    /// @param ptr start of the array.
    /// @param size length of the array.
    explicit owned_slice(const T* ptr, size_type size)
        : owned_slice { slice<T> { ptr, size } }
    {
    }

    /// @brief Constructs a new slice from an array.
    ///
    /// @tparam N size of the array.
    /// @param array array the slice points to.
    template <std::size_t N>
    explicit owned_slice(const T& array)
        : owned_slice {
            slice<T> { array }
        }
    {
    }

    /// Copies another slice.
    ///
    /// @param other other slice.
    owned_slice(const owned_slice& other)
        : owned_slice { slice<T> { other.data(), other.size() } }
    {
    }

    owned_slice(owned_slice&& other) noexcept = default;

    owned_slice& operator=(const owned_slice& rhs) = default;
    owned_slice& operator=(owned_slice&& rhs) = default;

    /// Checks that the elements of the two slices are equal.
    /// Is equivalent to calling std::equal with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the two slices are equal, `false` otherwise.
    template <typename U>
    bool operator==(const owned_slice<U>& other) const
    {
        return std::equal(this->begin(), this->end(), other.begin());
    }

    /// Checks that the elements of the two slices are equal.
    /// Is equivalent to negating the call to std::equal with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the two slices are not equal, `false` otherwise.
    template <typename U>
    bool operator!=(const owned_slice<U>& other) const
    {
        return !(*this == other);
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the contents of this are lexicographically less than
    /// the contents of other, `false` otherwise.
    template <typename U>
    bool operator<(const owned_slice<U>& other) const
    {
        return std::lexicographical_compare(
            this->begin(), this->end(),
            other.begin(), other.end());
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the contents of this are lexicographically less than
    /// or equal to the contents of other, `false` otherwise.
    template <typename U>
    bool operator<=(const owned_slice<U>& other) const
    {
        return std::lexicographical_compare(
            this->begin(), this->end(),
            other.begin(), other.end(),
            [](const T& lhs, const T& rhs) { return lhs <= rhs; });
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the contents of this are lexicographically greater than
    /// the contents of other, `false` otherwise.
    template <typename U>
    bool operator>(const owned_slice<U>& other) const
    {
        return std::lexicographical_compare(
            this->begin(), this->end(),
            other.begin(), other.end(),
            [](const T& lhs, const T& rhs) { return lhs > rhs; });
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the contents of this are lexicographically greater than
    /// or equal to the contents of other, `false` otherwise.
    template <typename U>
    bool operator>=(const owned_slice<U>& other) const
    {
        return std::lexicographical_compare(
            this->begin(), this->end(),
            other.begin(), other.end(),
            [](const T& lhs, const T& rhs) { return lhs >= rhs; });
    }

    /// Returns a reference to the element at specified location pos, with bounds checking.
    /// If pos is not within the range of the container, an exception of type std::out_of_range
    /// is thrown.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    reference at(size_type pos)
    {
        if (pos >= this->m_len)
            throw std::out_of_range("slice access out of range");
        return (*this)[pos];
    }

    /// Returns a reference to the element at specified location pos, with bounds checking.
    /// If pos is not within the range of the container, an exception of type std::out_of_range
    /// is thrown.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    const_reference at(size_type pos) const
    {
        if (pos >= this->m_len)
            throw std::out_of_range("slice access out of range");
        return (*this)[pos];
    }

    /// Returns a reference to the element at specified location pos.
    /// No bounds checking is performed.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    reference operator[](size_type pos)
    {
        return this->m_ptr[pos];
    }

    /// Returns a reference to the element at specified location pos.
    /// No bounds checking is performed.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    const_reference operator[](size_type pos) const
    {
        return this->m_ptr[pos];
    }

    /// Returns a reference to the first element in the container.
    /// @return Calling front on an empty container is undefined.
    reference front()
    {
        return *this->begin();
    }

    /// Returns a reference to the first element in the container.
    /// @return Calling front on an empty container is undefined.
    const_reference front() const
    {
        return *this->begin();
    }

    /// Returns a reference to the last element in the container.
    /// @return Calling back on an empty container is undefined.
    reference back()
    {
        return *std::prev(this->end());
    }

    /// Returns a reference to the last element in the container.
    /// @return Calling back on an empty container is undefined.
    const_reference back() const
    {
        return *std::prev(this->end());
    }

    /// Returns pointer to the underlying array serving as element storage.
    /// @return Pointer to the underlying element storage.
    T* data() noexcept
    {
        return this->m_ptr;
    }

    /// Returns pointer to the underlying array serving as element storage.
    /// @return Pointer to the underlying element storage.
    const T* data() const noexcept
    {
        return this->m_ptr;
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    iterator begin() noexcept
    {
        return iterator { this->m_ptr };
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    const_iterator begin() const noexcept
    {
        return const_iterator { this->m_ptr };
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    const_iterator cbegin() const noexcept
    {
        return const_iterator { this->m_ptr };
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    iterator end() noexcept
    {
        return iterator { this->m_ptr + this->m_len };
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    const_iterator end() const noexcept
    {
        return const_iterator { this->m_ptr + this->m_len };
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    const_iterator cend() const noexcept
    {
        return const_iterator { this->m_ptr + this->m_len };
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    reverse_iterator rbegin() noexcept
    {
        return reverse_iterator { this->end() };
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator { this->end() };
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator { this->cend() };
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    reverse_iterator rend() noexcept
    {
        return reverse_iterator { this->begin() };
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator { this->begin() };
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator { this->cbegin() };
    }

    /// Checks if the container has no elements, i.e. whether `begin() == end()`.
    ///
    /// @return `true` if the container is empty, `false` otherwise.
    [[nodiscard]] bool empty() const noexcept
    {
        return this->begin() == this->end();
    }

    /// Returns the number of elements in the container, i.e. `std::distance(begin(), end())`.
    ///
    /// @return The number of elements in the container.
    [[nodiscard]] size_type size() const noexcept
    {
        return std::distance(this->begin(), this->end());
    }

    /// Assigns the `value` to all elements in the container.
    ///
    /// @param value the value to assign to the elements
    void fill(const T& value)
    {
        std::fill(this->begin(), this->end(), value);
    }
};

#define OWNED_SLICE_EXTERN(T, N)                                                                                         \
    static_assert(std::is_standard_layout<slice<T>>::value, "slice<" #T "> must be a standard-layout type");             \
    static_assert(std::is_standard_layout<owned_slice<T>>::value, "owned_slice<" #T "> must be a standard-layout type"); \
    owned_slice<T> wavelet_rs_slice_##N##_new(slice<T>);                                                                 \
    void wavelet_rs_slice_##N##_free(owned_slice<T>);

#define OWNED_SLICE_SPEC(T, N)                                 \
    template <>                                                \
    owned_slice<T>::~owned_slice()                             \
    {                                                          \
        if (this->m_ptr != reinterpret_cast<T*>(alignof(T))) { \
            wavelet_rs_slice_##N##_free(std::move(*this));     \
        }                                                      \
    }                                                          \
    template <>                                                \
    owned_slice<T>::owned_slice(slice<T> s)                    \
        : m_ptr { reinterpret_cast<T*>(alignof(T)) }           \
        , m_len { 0 }                                          \
    {                                                          \
        auto tmp { wavelet_rs_slice_##N##_new(s) };            \
        std::swap(*this, tmp);                                 \
    }

extern "C" {
// NOLINTBEGIN(*-return-type-c-linkage)
OWNED_SLICE_EXTERN(char, c_char)
OWNED_SLICE_EXTERN(std::uint8_t, u8)
OWNED_SLICE_EXTERN(std::uint16_t, u16)
OWNED_SLICE_EXTERN(std::uint32_t, u32)
OWNED_SLICE_EXTERN(std::uint64_t, u64)
OWNED_SLICE_EXTERN(std::int8_t, i8)
OWNED_SLICE_EXTERN(std::int16_t, i16)
OWNED_SLICE_EXTERN(std::int32_t, i32)
OWNED_SLICE_EXTERN(std::int64_t, i64)
OWNED_SLICE_EXTERN(float, f32)
OWNED_SLICE_EXTERN(double, f64)
// NOLINTEND(*-return-type-c-linkage)
}

OWNED_SLICE_SPEC(char, c_char)
OWNED_SLICE_SPEC(std::uint8_t, u8)
OWNED_SLICE_SPEC(std::uint16_t, u16)
OWNED_SLICE_SPEC(std::uint32_t, u32)
OWNED_SLICE_SPEC(std::uint64_t, u64)
OWNED_SLICE_SPEC(std::int8_t, i8)
OWNED_SLICE_SPEC(std::int16_t, i16)
OWNED_SLICE_SPEC(std::int32_t, i32)
OWNED_SLICE_SPEC(std::int64_t, i64)
OWNED_SLICE_SPEC(float, f32)
OWNED_SLICE_SPEC(double, f64)

/// Owned string type.
class string {
    owned_slice<char> m_buff;

public:
    typedef char CharT;
    typedef owned_slice<char>::value_type value_type;
    typedef owned_slice<char>::size_type size_type;
    typedef owned_slice<char>::difference_type difference_type;
    typedef owned_slice<char>::reference reference;
    typedef owned_slice<char>::const_reference const_reference;
    typedef owned_slice<char>::pointer pointer;
    typedef owned_slice<char>::const_pointer const_pointer;
    typedef owned_slice<char>::iterator iterator;
    typedef owned_slice<char>::const_iterator const_iterator;
    typedef owned_slice<char>::reverse_iterator reverse_iterator;
    typedef owned_slice<char>::const_reverse_iterator const_reverse_iterator;

    explicit string() = default;
    explicit string(const CharT* s, size_type count)
        : m_buff { slice<char> { s, count } }
    {
    }
    explicit string(const CharT* s)
        : string { s, std::strlen(s) }
    {
    }

    string(const string& other) = default;
    string(string&& other) noexcept = default;
    ~string() = default;

    string& operator=(const string& rhs) = default;
    string& operator=(string&& rhs) = default;

    /// Checks that the elements of the two strings are equal.
    /// Is equivalent to calling std::equal with the two strings.
    ///
    /// @param other other string.
    /// @return `true` if the two strings are equal, `false` otherwise.
    bool operator==(const string& other) const
    {
        return this->m_buff == other.m_buff;
    }

    /// Checks that the elements of the two strings are equal.
    /// Is equivalent to negating the call to std::equal with the two strings.
    ///
    /// @param other other string.
    /// @return `true` if the two strings are not equal, `false` otherwise.
    bool operator!=(const string& other) const
    {
        return this->m_buff != other.m_buff;
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two strings.
    ///
    /// @param other other string.
    /// @return `true` if the contents of this are lexicographically less than
    /// the contents of other, `false` otherwise.
    bool operator<(const string& other) const
    {
        return this->m_buff < other.m_buff;
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two strings.
    ///
    /// @param other other string.
    /// @return `true` if the contents of this are lexicographically less than
    /// or equal to the contents of other, `false` otherwise.
    bool operator<=(const string& other) const
    {
        return this->m_buff <= other.m_buff;
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two strings.
    ///
    /// @param other other string.
    /// @return `true` if the contents of this are lexicographically greater than
    /// the contents of other, `false` otherwise.
    bool operator>(const string& other) const
    {
        return this->m_buff > other.m_buff;
    }

    /// Compares the contents of lhs and rhs lexicographically.
    /// Is equivalent to calling std::lexicographical_compare with the two strings.
    ///
    /// @param other other string.
    /// @return `true` if the contents of this are lexicographically greater than
    /// or equal to the contents of other, `false` otherwise.
    bool operator>=(const string& other) const
    {
        return this->m_buff >= other.m_buff;
    }

    /// Returns a reference to the element at specified location pos, with bounds checking.
    /// If pos is not within the range of the container, an exception of type std::out_of_range
    /// is thrown.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    reference at(size_type pos)
    {
        return this->m_buff.at(pos);
    }

    /// Returns a reference to the element at specified location pos, with bounds checking.
    /// If pos is not within the range of the container, an exception of type std::out_of_range
    /// is thrown.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    const_reference at(size_type pos) const
    {
        return this->m_buff.at(pos);
    }

    /// Returns a reference to the element at specified location pos.
    /// No bounds checking is performed.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    reference operator[](size_type pos)
    {
        return this->m_buff[pos];
    }

    /// Returns a reference to the element at specified location pos.
    /// No bounds checking is performed.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    const_reference operator[](size_type pos) const
    {
        return this->m_buff[pos];
    }

    /// Returns a reference to the first element in the container.
    /// @return Calling front on an empty container is undefined.
    reference front()
    {
        return *this->begin();
    }

    /// Returns a reference to the first element in the container.
    /// @return Calling front on an empty container is undefined.
    const_reference front() const
    {
        return *this->begin();
    }

    /// Returns a reference to the last element in the container.
    /// @return Calling back on an empty container is undefined.
    reference back()
    {
        return *std::prev(this->end());
    }

    /// Returns a reference to the last element in the container.
    /// @return Calling back on an empty container is undefined.
    const_reference back() const
    {
        return *std::prev(this->end());
    }

    /// Returns pointer to the underlying array serving as element storage.
    /// @return Pointer to the underlying element storage.
    CharT* data() noexcept
    {
        return this->m_buff.data();
    }

    /// Returns pointer to the underlying array serving as element storage.
    /// @return Pointer to the underlying element storage.
    const CharT* data() const noexcept
    {
        return this->m_buff.data();
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    iterator begin() noexcept
    {
        return this->m_buff.begin();
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    const_iterator begin() const noexcept
    {
        return this->m_buff.begin();
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    const_iterator cbegin() const noexcept
    {
        return this->m_buff.cbegin();
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    iterator end() noexcept
    {
        return this->m_buff.end();
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    const_iterator end() const noexcept
    {
        return this->m_buff.end();
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    const_iterator cend() const noexcept
    {
        return this->m_buff.cend();
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    reverse_iterator rbegin() noexcept
    {
        return this->m_buff.rbegin();
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    const_reverse_iterator rbegin() const noexcept
    {
        return this->m_buff.rbegin();
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    const_reverse_iterator crbegin() const noexcept
    {
        return this->m_buff.crbegin();
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    reverse_iterator rend() noexcept
    {
        return this->m_buff.rend();
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    const_reverse_iterator rend() const noexcept
    {
        return this->m_buff.rend();
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    const_reverse_iterator crend() const noexcept
    {
        return this->m_buff.crend();
    }

    /// Checks if the container has no elements, i.e. whether `begin() == end()`.
    ///
    /// @return `true` if the container is empty, `false` otherwise.
    [[nodiscard]] bool empty() const noexcept
    {
        return this->m_buff.empty();
    }

    /// Returns the number of elements in the container, i.e. `std::distance(begin(), end())`.
    ///
    /// @return The number of elements in the container.
    [[nodiscard]] size_type size() const noexcept
    {
        return this->m_buff.size();
    }
};

template <typename T>
class option {
    std::int8_t m_tag;
    union option_impl {
        T some;

        option_impl() noexcept { }
        ~option_impl() noexcept { }
    } m_data;

    static constexpr std::int8_t NONE_TAG = 1;
    static constexpr std::int8_t SOME_TAG = 1;

public:
    typedef T value_type;

    /// Constructs an empty option.
    option() noexcept
        : m_tag { NONE_TAG }
    {
    }

    /// Constructs an option with a value.
    ///
    /// @param value value to be contained in the option.
    option(T value) noexcept(
        std::is_nothrow_move_constructible<T>::value)
        : m_tag { SOME_TAG }
    {
        new (&(this->m_data.some)) T { std::move(value) };
    }

    /// Copy constructor.
    ///
    /// @param other other option.
    option(const option& other) noexcept(
        std::is_nothrow_copy_constructible<T>::value)
        : m_tag { other.m_tag }
        , m_data {}
    {
        if (other.has_value()) {
            new (&(this->m_data.some)) T { *other };
        }
    }

    /// Move constructor.
    ///
    /// @param other other option.
    option(option&& other) noexcept = default;

    ~option() noexcept(std::is_nothrow_destructible<T>::value)
    {
        if (this->has_value()) {
            (**this).~T();
        }
    }

    /// Copy assignment.
    ///
    /// @param other other option.
    option& operator=(const option& other) noexcept(
        std::is_nothrow_destructible<T>::value&&
            std::is_nothrow_copy_assignable<T>::value&&
                std::is_nothrow_copy_constructible<T>::value)
    {
        if (this != &other) {
            if (this->has_value()) {
                if (other.has_value()) {
                    **this = *other;
                } else {
                    (**this).~T();
                }
            } else {
                if (other.has_value()) {
                    new (&(this->m_data.some)) T { *other };
                }
            }

            this->m_tag = other.m_tag;
        }

        return *this;
    }

    /// Move assignment.
    ///
    /// @param other other option.
    option& operator=(option&& other) noexcept(
        std::is_nothrow_destructible<T>::value&&
            std::is_nothrow_move_assignable<T>::value)
    {
        if (this != &other) {
            if (this->has_value()) {
                if (other.has_value()) {
                    **this = std::move(*other);
                } else {
                    (**this).~T();
                }
            } else {
                if (other.has_value()) {
                    new (&(this->m_data.some)) T { std::move(*other) };
                }
            }

            this->m_tag = other.m_tag;
        }

        return *this;
    }

    /// Returns a pointer to the contained value.
    /// Dereferencing the pointer of an empty option results in undefined behaviour.
    ///
    /// @return pointer to the contained element.
    T* operator->() noexcept
    {
        return &this->m_data.some;
    }

    /// Returns a pointer to the contained value.
    /// Dereferencing the pointer of an empty option results in undefined behaviour.
    ///
    /// @return pointer to the contained element.
    const T* operator->() const noexcept
    {
        return &this->m_data.some;
    }

    /// Returns a reference to the contained value.
    /// Accessing the element of an empty option results in undefined behaviour.
    ///
    /// @return reference to the contained element.
    T&
    operator*() & noexcept
    {
        return this->m_data.some;
    }

    /// Returns a reference to the contained value.
    /// Accessing the element of an empty option results in undefined behaviour.
    ///
    /// @return reference to the contained element.
    const T& operator*() const& noexcept
    {
        return this->m_data.some;
    }

    /// Returns a reference to the contained value.
    /// Accessing the element of an empty option results in undefined behaviour.
    ///
    /// @return reference to the contained element.
    T&&
    operator*() && noexcept
    {
        return std::move(this->m_data.some);
    }

    /// Returns a reference to the contained value.
    /// Accessing the element of an empty option results in undefined behaviour.
    ///
    /// @return reference to the contained element.
    const T&& operator*() const&& noexcept
    {
        return std::move(this->m_data.some);
    }

    /// Returns whether the option contains a value
    ///
    /// @return `true` if the option contains a value, `false` otherwise.
    operator bool() const noexcept
    {
        return this->has_value();
    }

    /// Returns whether the option contains a value
    ///
    /// @return `true` if the option contains a value, `false` otherwise.
    bool has_value() const noexcept
    {
        return this->m_tag == SOME_TAG;
    }

    /// Returns a reference to the contained value.
    ///
    /// @return reference to the contained element.
    T& value() & noexcept
    {
        assert(this->has_value());
        return this->m_data.some;
    }

    /// Returns a reference to the contained value.
    ///
    /// @return reference to the contained element.
    const T& value() const& noexcept
    {
        assert(this->has_value());
        return this->m_data.some;
    }

    /// Returns a reference to the contained value.
    ///
    /// @return reference to the contained element.
    T&& value() && noexcept
    {
        assert(this->has_value());
        return std::move(this->m_data.some);
    }

    /// Returns a reference to the contained value.
    ///
    /// @return reference to the contained element.
    const T&& value() const&& noexcept
    {
        assert(this->has_value());
        return std::move(this->m_data.some);
    }
};

/// Marker for callables that can only be invoked once.
struct FnOnce { };

/// Marker for callables that can only be invoked mutably.
struct FnMut { };

/// Marker for callables that can be invoked immutably.
struct Fn { };

template <typename T, typename Ret, typename... Args>
class callable {
public:
    typedef Ret (*function_type)(Args...);

private:
    static_assert(std::is_same<T, FnOnce>::value
            || std::is_same<T, FnMut>::value
            || std::is_same<T, Fn>::value,
        "invalid type parameter");

    union ctx_t {
        void* allocated;
        function_type func_ptr;
    };
    typedef void (*drop_func_t)(ctx_t);
    typedef Ret (*call_func_t)(ctx_t, Args...);

    static void empty_drop(ctx_t) { }
    [[noreturn]] static Ret empty_call(ctx_t, Args...) { assert(false); }
    static Ret fn_ptr_call(ctx_t ctx, Args... args) { return ctx.func_ptr(std::move(args)...); }

    ctx_t m_ctx;
    drop_func_t m_drop;
    call_func_t m_call;

public:
    /// Constructs a new instance using a function pointer.
    ///
    /// @param f callable function pointer.
    explicit callable(function_type f) noexcept
        : m_ctx { .func_ptr = f }
        , m_drop { empty_drop }
        , m_call { fn_ptr_call }
    {
    }

    /// Constructs a new instance using the provided callable.
    ///
    /// @tparam F Type of the callable.
    /// @param f instance of the callable.
    template <typename F, typename T2 = T>
    explicit callable(F&& f, typename std::enable_if<std::is_same<T2, FnOnce>::value>::type* = nullptr)
        : m_ctx { .allocated = nullptr }
        , m_drop { nullptr }
        , m_call { nullptr }
    {
        using F_ = typename std::remove_reference<typename std::remove_cv<F>::type>::type;

        F_* ptr = new F_ { std::forward<F>(f) };
        this->m_ctx.allocated = static_cast<void*>(ptr);
        this->m_drop = +[](ctx_t ctx) {
            F_* ptr = static_cast<F_*>(ctx.allocated);
            delete ptr;
        };
        this->m_call = +[](ctx_t ctx, Args... args) -> auto
        {
            F_* ptr = static_cast<F_*>(ctx.allocated);
            auto res = (*ptr)(std::move(args)...);
            delete ptr;
            return res;
        };
    }

    /// Constructs a new instance using the provided callable.
    ///
    /// @tparam F Type of the callable.
    /// @param f instance of the callable.
    template <typename F, typename T2 = T>
    explicit callable(F&& f, typename std::enable_if<std::is_same<T2, FnMut>::value>::type* = nullptr)
        : m_ctx { .allocated = nullptr }
        , m_drop { nullptr }
        , m_call { nullptr }
    {
        using F_ = typename std::remove_reference<typename std::remove_cv<F>::type>::type;

        F_* ptr = new F_ { std::forward<F>(f) };
        this->m_ctx.allocated = static_cast<void*>(ptr);
        this->m_drop = +[](ctx_t ctx) {
            F_* ptr = static_cast<F_*>(ctx.allocated);
            delete ptr;
        };
        this->m_call = +[](ctx_t ctx, Args... args) -> auto
        {
            F_* ptr = static_cast<F_*>(ctx.allocated);
            return (*ptr)(std::move(args)...);
        };
    }

    /// Constructs a new instance using the provided callable.
    ///
    /// @tparam F Type of the callable.
    /// @param f instance of the callable.
    template <typename F, typename T2 = T>
    explicit callable(F&& f, typename std::enable_if<std::is_same<T2, Fn>::value>::type* = nullptr)
        : m_ctx { .allocated = nullptr }
        , m_drop { nullptr }
        , m_call { nullptr }
    {
        using F_ = typename std::remove_reference<typename std::remove_cv<F>::type>::type;

        F_* ptr = new F_ { std::forward<F>(f) };
        this->m_ctx.allocated = static_cast<void*>(ptr);
        this->m_drop = +[](ctx_t ctx) {
            F_* ptr = static_cast<F_*>(ctx.allocated);
            delete ptr;
        };
        this->m_call = +[](ctx_t ctx, Args... args) -> auto
        {
            const F_* ptr = static_cast<const F_*>(ctx.allocated);
            return (*ptr)(std::move(args)...);
        };
    }

    callable(const callable& other) = delete;

    callable(callable&& other) noexcept
        : m_ctx { other.m_ctx }
        , m_drop { std::exchange(other.m_drop, empty_drop) }
        , m_call { std::exchange(other.m_call, empty_call) }
    {
    }

    ~callable()
    {
        if (this->m_drop != empty_drop) {
            this->m_drop(this->m_ctx);
        }

        // fix clang-tidy false positive.
        this->m_ctx.~ctx_t();
    }

    callable& operator=(const callable& rhs) = delete;

    callable& operator=(callable&& rhs) noexcept
    {
        if (this != &rhs) {
            std::swap(this->m_ctx, rhs.m_ctx);
            std::swap(this->m_drop, rhs.m_drop);
            std::swap(this->m_call, rhs.m_call);
        }

        return *this;
    }

    template <typename T2 = T>
    typename std::enable_if<std::is_same<T2, FnOnce>::value, Ret>::type
    operator()(Args&&... args)
    {
        auto res = this->m_call(this->m_ctx, std::forward<Args>(args)...);

        // fix clang-tidy false positive.
        this->m_ctx.~ctx_t();

        this->m_ctx.allocated = nullptr;
        this->m_drop = empty_drop;
        this->m_call = empty_call;

        return res;
    }

    template <typename T2 = T>
    typename std::enable_if<std::is_same<T2, FnMut>::value
            || std::is_same<T2, Fn>::value,
        Ret>::type
    operator()(Args&&... args)
    {
        return this->m_call(this->m_ctx, std::forward<Args>(args)...);
    }

    template <typename T2 = T>
    typename std::enable_if<std::is_same<T2, Fn>::value, Ret>::type
    operator()(Args&&... args) const
    {
        return this->m_call(this->m_ctx, std::forward<Args>(args)...);
    }
};

/// Callable reading an element from the provided position.
/// The callable must be thread-safe.
///
/// @tparam T element type of the volume.
template <typename T>
using volume_fetcher = callable<Fn, T, slice<std::size_t>>;

/// Callable reading an element from the provided position in the current block.
///
/// @tparam T element type of the volume.
template <typename T>
using block_reader = callable<Fn, T, slice<std::size_t>>;

/// Callable returning a reader for a volume block.
/// The callable must be thread-safe.
///
/// @tparam T element type of the volume.
template <typename T>
using block_reader_fetcher = callable<Fn, block_reader<T>, std::size_t>;

/// Callable returning a fetcher for volume block readers.
///
/// @tparam T element type of the volume.
template <typename T>
using reader_fetcher = callable<FnOnce, block_reader_fetcher<T>, slice<std::size_t>>;

/// Callable writing an element at the provided position in the current block.
///
/// @tparam T element type of the volume.
template <typename T>
using block_writer = callable<FnMut, void, slice<std::size_t>, T>;

/// Callable returning a writer for a volume block.
/// The callable must be thread-safe.
///
/// @tparam T element type of the volume.
template <typename T>
using block_writer_fetcher = callable<Fn, block_writer<T>, std::size_t>;

/// Callable returning a fetcher for volume block writers.
///
/// @tparam T element type of the volume.
template <typename T>
using writer_fetcher = callable<FnOnce, block_writer_fetcher<T>, slice<std::size_t>>;

///////////////// Encoder /////////////////

template <typename T>
class encoder {
public:
    class encoder_impl;

private:
    encoder_impl* m_enc;

public:
    /// Constructs a new encoder spanning a volume with dimmensions `dims`.
    ///
    /// @param dims dimmensions of the volume to be encoded.
    /// @param num_base_dims number of dimmensions contained in each volume_fetcher.
    encoder(slice<std::size_t> dims, std::size_t num_base_dims) = delete;
    encoder(const encoder& other) = delete;

    encoder(encoder&& other) noexcept
        : m_enc { std::exchange(other.m_enc, nullptr) }
    {
    }

    ~encoder() = delete;

    encoder& operator=(const encoder& other) = delete;

    encoder& operator=(encoder&& other) noexcept
    {
        if (this != &other) {
            std::swap(this->m_enc, other.m_enc);
        }

        return *this;
    }

    /// Inserts a volume_fetcher into the encoder at the position `index`.
    /// The position `index` must not be populated and be in the range [dims[num_base_dims],...].
    ///
    /// @param index position of the fetcher in the volume.
    /// @param fetcher fetcher to be inserted.
    void add_fetcher(slice<std::size_t> index, volume_fetcher<T> fetcher) = delete;

    /// Encodes the volume using the inserted fetchers and the specified filter.
    /// The output will be written into the `output` directory. The caller must
    /// ensure that `output` exists and is a writable directory. The selected
    /// block size must tile the input volume exactly.
    ///
    /// @tparam Filter filter to use for the encoding.
    /// @param output output directory.
    /// @param block_size selected block size.
    template <typename Filter>
    void encode(const char* output, slice<std::size_t> block_size) const = delete;

    /// Fetches an element from the encoder metadata.
    ///
    /// @tparam U type of the element.
    /// @param key key of the mapped element.
    /// @return metadata element.
    template <typename U>
    option<U> metadata_get(const char* key) const = delete;

    /// Inserts an element into the encoder metadata.
    ///
    /// @tparam U type of the element.
    /// @param key key where the element will be mapped to.
    /// @param value element to insert into the metadata.
    /// @return `true` if an old value was overwritten, `false` otherwise.
    template <typename U>
    bool metadata_insert(const char* key, U value) = delete;
};

struct HaarWavelet;
struct AverageFilter;

template <typename T>
using array_1 = std::array<T, 1>;
template <typename T>
using array_2 = std::array<T, 2>;
template <typename T>
using array_3 = std::array<T, 3>;
template <typename T>
using array_4 = std::array<T, 4>;

#define ENCODER_METADATA_EXTERN_(T, N, M, MN)                                                                  \
    static_assert(std::is_standard_layout<M>::value, #M " must be a standard-layout type");                    \
    static_assert(std::is_standard_layout<option<M>>::value, "option<" #M "> must be a standard-layout type"); \
    option<M> wavelet_rs_encoder_##N##_metadata_get_##MN(encoder<T>::encoder_impl*, const char*);              \
    std::uint8_t wavelet_rs_encoder_##N##_metadata_insert_##MN(encoder<T>::encoder_impl*, const char*, M);

#ifdef WAVELET_RS_IMPORT_MEATADATA_ARR
#define ENCODER_METADATA_ARRAY_EXTERN(T, N, M, MN)         \
    ENCODER_METADATA_EXTERN_(T, N, array_1<M>, MN##_arr_1) \
    ENCODER_METADATA_EXTERN_(T, N, array_2<M>, MN##_arr_2) \
    ENCODER_METADATA_EXTERN_(T, N, array_3<M>, MN##_arr_3) \
    ENCODER_METADATA_EXTERN_(T, N, array_4<M>, MN##_arr_4)
#else
#define ENCODER_METADATA_ARRAY_EXTERN(T, N, M, MN)
#endif // WAVELET_RS_IMPORT_MEATADATA_ARR

#ifdef WAVELET_RS_IMPORT_MEATADATA_SLICE
#define ENCODER_METADATA_SLICE_EXTERN(T, N, M, MN) \
    ENCODER_METADATA_EXTERN_(T, N, owned_slice<M>, MN##_slice)
#else
#define ENCODER_METADATA_SLICE_EXTERN(T, N, M, MN)
#endif // WAVELET_RS_IMPORT_MEATADATA_SLICE

#define ENCODER_METADATA_EXTERN(T, N, M, MN)   \
    ENCODER_METADATA_EXTERN_(T, N, M, MN)      \
    ENCODER_METADATA_ARRAY_EXTERN(T, N, M, MN) \
    ENCODER_METADATA_SLICE_EXTERN(T, N, M, MN)

#define ENCODER_EXTERN_(T, N)                                                                                                  \
    static_assert(std::is_standard_layout<slice<std::size_t>>::value, "slice<std::size_t> must be a standard-layout type");    \
    static_assert(std::is_standard_layout<volume_fetcher<T>>::value, "volume_fetcher<" #T "> must be a standard-layout type"); \
    encoder<T>::encoder_impl* wavelet_rs_encoder_##N##_new(slice<std::size_t>, std::size_t);                                   \
    void wavelet_rs_encoder_##N##_free(encoder<T>::encoder_impl*);                                                             \
    void wavelet_rs_encoder_##N##_add_fetcher(encoder<T>::encoder_impl*, slice<std::size_t>, volume_fetcher<T>);               \
    void wavelet_rs_encoder_##N##_encode_haar(const encoder<T>::encoder_impl*, const char*, slice<std::size_t>);               \
    void wavelet_rs_encoder_##N##_encode_average(const encoder<T>::encoder_impl*, const char*, slice<std::size_t>);            \
    ENCODER_METADATA_EXTERN(T, N, std::uint8_t, u8)                                                                            \
    ENCODER_METADATA_EXTERN(T, N, std::uint16_t, u16)                                                                          \
    ENCODER_METADATA_EXTERN(T, N, std::uint32_t, u32)                                                                          \
    ENCODER_METADATA_EXTERN(T, N, std::uint64_t, u64)                                                                          \
    ENCODER_METADATA_EXTERN(T, N, std::int8_t, i8)                                                                             \
    ENCODER_METADATA_EXTERN(T, N, std::int16_t, i16)                                                                           \
    ENCODER_METADATA_EXTERN(T, N, std::int32_t, i32)                                                                           \
    ENCODER_METADATA_EXTERN(T, N, std::int64_t, i64)                                                                           \
    ENCODER_METADATA_EXTERN(T, N, float, f32)                                                                                  \
    ENCODER_METADATA_EXTERN(T, N, double, f64)                                                                                 \
    ENCODER_METADATA_EXTERN_(T, N, string, string)

#ifdef WAVELET_RS_IMPORT_VEC
#define ENCODER_VEC_EXTERN(T, N)           \
    ENCODER_EXTERN_(array_1<T>, vec_1_##N) \
    ENCODER_EXTERN_(array_2<T>, vec_2_##N) \
    ENCODER_EXTERN_(array_3<T>, vec_3_##N) \
    ENCODER_EXTERN_(array_4<T>, vec_4_##N)
#else
#define ENCODER_VEC_EXTERN(T, N)
#endif // WAVELET_RS_IMPORT_VEC

#ifdef WAVELET_RS_IMPORT_MAT
#define ENCODER_MATRIX_EXTERN(T, N)                   \
    ENCODER_EXTERN_(array_1<array_1<T>>, mat_1x1_##N) \
    ENCODER_EXTERN_(array_1<array_2<T>>, mat_1x2_##N) \
    ENCODER_EXTERN_(array_1<array_3<T>>, mat_1x3_##N) \
    ENCODER_EXTERN_(array_1<array_4<T>>, mat_1x4_##N) \
    ENCODER_EXTERN_(array_2<array_1<T>>, mat_2x1_##N) \
    ENCODER_EXTERN_(array_2<array_2<T>>, mat_2x2_##N) \
    ENCODER_EXTERN_(array_2<array_3<T>>, mat_2x3_##N) \
    ENCODER_EXTERN_(array_2<array_4<T>>, mat_2x4_##N) \
    ENCODER_EXTERN_(array_3<array_1<T>>, mat_3x1_##N) \
    ENCODER_EXTERN_(array_3<array_2<T>>, mat_3x2_##N) \
    ENCODER_EXTERN_(array_3<array_3<T>>, mat_3x3_##N) \
    ENCODER_EXTERN_(array_3<array_4<T>>, mat_3x4_##N) \
    ENCODER_EXTERN_(array_4<array_1<T>>, mat_4x1_##N) \
    ENCODER_EXTERN_(array_4<array_2<T>>, mat_4x2_##N) \
    ENCODER_EXTERN_(array_4<array_3<T>>, mat_4x3_##N) \
    ENCODER_EXTERN_(array_4<array_4<T>>, mat_4x4_##N)
#else
#define ENCODER_MATRIX_EXTERN(T, N)
#endif // WAVELET_RS_IMPORT_MAT

#define ENCODER_EXTERN(T, N) \
    ENCODER_EXTERN_(T, N)    \
    ENCODER_VEC_EXTERN(T, N) \
    ENCODER_MATRIX_EXTERN(T, N)

extern "C" {
// NOLINTBEGIN(*-return-type-c-linkage)
ENCODER_EXTERN(float, f32)
ENCODER_EXTERN(double, f64)
// NOLINTEND(*-return-type-c-linkage)
}

#define ENCODER_METADATA_SPEC_(T, N, M, MN)                                                       \
    template <>                                                                                   \
    template <>                                                                                   \
    option<M> encoder<T>::metadata_get<M>(const char* key) const                                  \
    {                                                                                             \
        return wavelet_rs_encoder_##N##_metadata_get_##MN(this->m_enc, key);                      \
    }                                                                                             \
    template <>                                                                                   \
    template <>                                                                                   \
    bool encoder<T>::metadata_insert<M>(const char* key, M value)                                 \
    {                                                                                             \
        return wavelet_rs_encoder_##N##_metadata_insert_##MN(this->m_enc, key, std::move(value)); \
    }

#ifdef WAVELET_RS_IMPORT_MEATADATA_ARR
#define ENCODER_METADATA_ARRAY_SPEC(T, N, M, MN)         \
    ENCODER_METADATA_SPEC_(T, N, array_1<M>, MN##_arr_1) \
    ENCODER_METADATA_SPEC_(T, N, array_2<M>, MN##_arr_2) \
    ENCODER_METADATA_SPEC_(T, N, array_3<M>, MN##_arr_3) \
    ENCODER_METADATA_SPEC_(T, N, array_4<M>, MN##_arr_4)
#else
#define ENCODER_METADATA_ARRAY_SPEC(T, N, M, MN)
#endif // WAVELET_RS_IMPORT_MEATADATA_ARR

#ifdef WAVELET_RS_IMPORT_MEATADATA_SLICE
#define ENCODER_METADATA_SLICE_SPEC(T, N, M, MN) \
    ENCODER_METADATA_SPEC_(T, N, owned_slice<M>, MN##_slice)
#else
#define ENCODER_METADATA_SLICE_SPEC(T, N, M, MN)
#endif // WAVELET_RS_IMPORT_MEATADATA_SLICE

#define ENCODER_METADATA_SPEC(T, N, M, MN)   \
    ENCODER_METADATA_SPEC_(T, N, M, MN)      \
    ENCODER_METADATA_ARRAY_SPEC(T, N, M, MN) \
    ENCODER_METADATA_SLICE_SPEC(T, N, M, MN)

#define ENCODER_SPEC_(T, N)                                                                         \
    template <>                                                                                     \
    encoder<T>::encoder(slice<std::size_t> dims, std::size_t num_base_dims)                         \
        : m_enc { wavelet_rs_encoder_##N##_new(dims, num_base_dims) }                               \
    {                                                                                               \
    }                                                                                               \
    template <>                                                                                     \
    encoder<T>::~encoder()                                                                          \
    {                                                                                               \
        if (this->m_enc != nullptr)                                                                 \
            wavelet_rs_encoder_##N##_free(this->m_enc);                                             \
    }                                                                                               \
    template <>                                                                                     \
    void encoder<T>::add_fetcher(slice<std::size_t> index, volume_fetcher<T> fetcher)               \
    {                                                                                               \
        wavelet_rs_encoder_##N##_add_fetcher(this->m_enc, index, std::move(fetcher));               \
    }                                                                                               \
    template <>                                                                                     \
    template <>                                                                                     \
    void encoder<T>::encode<HaarWavelet>(const char* output, slice<std::size_t> block_size) const   \
    {                                                                                               \
        wavelet_rs_encoder_##N##_encode_haar(this->m_enc, output, std::move(block_size));           \
    }                                                                                               \
    template <>                                                                                     \
    template <>                                                                                     \
    void encoder<T>::encode<AverageFilter>(const char* output, slice<std::size_t> block_size) const \
    {                                                                                               \
        wavelet_rs_encoder_##N##_encode_average(this->m_enc, output, std::move(block_size));        \
    }                                                                                               \
    ENCODER_METADATA_SPEC(T, N, std::uint8_t, u8)                                                   \
    ENCODER_METADATA_SPEC(T, N, std::uint16_t, u16)                                                 \
    ENCODER_METADATA_SPEC(T, N, std::uint32_t, u32)                                                 \
    ENCODER_METADATA_SPEC(T, N, std::uint64_t, u64)                                                 \
    ENCODER_METADATA_SPEC(T, N, std::int8_t, i8)                                                    \
    ENCODER_METADATA_SPEC(T, N, std::int16_t, i16)                                                  \
    ENCODER_METADATA_SPEC(T, N, std::int32_t, i32)                                                  \
    ENCODER_METADATA_SPEC(T, N, std::int64_t, i64)                                                  \
    ENCODER_METADATA_SPEC(T, N, float, f32)                                                         \
    ENCODER_METADATA_SPEC(T, N, double, f64)                                                        \
    ENCODER_METADATA_SPEC_(T, N, string, string)

#ifdef WAVELET_RS_IMPORT_VEC
#define ENCODER_VEC_SPEC_(T, N)          \
    ENCODER_SPEC_(array_1<T>, vec_1_##N) \
    ENCODER_SPEC_(array_2<T>, vec_2_##N) \
    ENCODER_SPEC_(array_3<T>, vec_3_##N) \
    ENCODER_SPEC_(array_4<T>, vec_4_##N)
#else
#define ENCODER_VEC_SPEC_(T, N)
#endif // WAVELET_RS_IMPORT_VEC

#ifdef WAVELET_RS_IMPORT_MAT
#define ENCODER_MATRIX_SPEC_(T, N)                  \
    ENCODER_SPEC_(array_1<array_1<T>>, mat_1x1_##N) \
    ENCODER_SPEC_(array_1<array_2<T>>, mat_1x2_##N) \
    ENCODER_SPEC_(array_1<array_3<T>>, mat_1x3_##N) \
    ENCODER_SPEC_(array_1<array_4<T>>, mat_1x4_##N) \
    ENCODER_SPEC_(array_2<array_1<T>>, mat_2x1_##N) \
    ENCODER_SPEC_(array_2<array_2<T>>, mat_2x2_##N) \
    ENCODER_SPEC_(array_2<array_3<T>>, mat_2x3_##N) \
    ENCODER_SPEC_(array_2<array_4<T>>, mat_2x4_##N) \
    ENCODER_SPEC_(array_3<array_1<T>>, mat_3x1_##N) \
    ENCODER_SPEC_(array_3<array_2<T>>, mat_3x2_##N) \
    ENCODER_SPEC_(array_3<array_3<T>>, mat_3x3_##N) \
    ENCODER_SPEC_(array_3<array_4<T>>, mat_3x4_##N) \
    ENCODER_SPEC_(array_4<array_1<T>>, mat_4x1_##N) \
    ENCODER_SPEC_(array_4<array_2<T>>, mat_4x2_##N) \
    ENCODER_SPEC_(array_4<array_3<T>>, mat_4x3_##N) \
    ENCODER_SPEC_(array_4<array_4<T>>, mat_4x4_##N)
#else
#define ENCODER_MATRIX_SPEC_(T, N)
#endif // WAVELET_RS_IMPORT_MAT

#define ENCODER_SPEC(T, N)  \
    ENCODER_SPEC_(T, N)     \
    ENCODER_VEC_SPEC_(T, N) \
    ENCODER_MATRIX_SPEC_(T, N)

ENCODER_SPEC(float, f32)
ENCODER_SPEC(double, f64)

///////////////// Decoder /////////////////

namespace dec_priv_ {
    class decoder_;

    template <typename T, typename F>
    struct decoder_impl {
        static constexpr bool implemented = false;
    };

    template <typename T, typename F, typename U>
    struct decoder_metadata_impl {
        static constexpr bool implemented = false;
    };

#define DECODER_METADATA_EXTERN_(T, F, N, M, MN)                                                               \
    static_assert(std::is_standard_layout<M>::value, #M " must be a standard-layout type");                    \
    static_assert(std::is_standard_layout<option<M>>::value, "option<" #M "> must be a standard-layout type"); \
    extern "C" option<M> wavelet_rs_decoder_##N##_metadata_get_##MN(decoder_*, const char*);                   \
    template <>                                                                                                \
    struct decoder_metadata_impl<T, F, M> {                                                                    \
        static constexpr bool implemented = true;                                                              \
        static constexpr auto get_fn = wavelet_rs_decoder_##N##_metadata_get_##MN;                             \
    };

#ifdef WAVELET_RS_IMPORT_MEATADATA_ARR
#define DECODER_METADATA_ARRAY_EXTERN(T, F, N, M, MN)         \
    DECODER_METADATA_EXTERN_(T, F, N, array_1<M>, MN##_arr_1) \
    DECODER_METADATA_EXTERN_(T, F, N, array_2<M>, MN##_arr_2) \
    DECODER_METADATA_EXTERN_(T, F, N, array_3<M>, MN##_arr_3) \
    DECODER_METADATA_EXTERN_(T, F, N, array_4<M>, MN##_arr_4)
#else
#define DECODER_METADATA_ARRAY_EXTERN(T, F, N, M, MN)
#endif // WAVELET_RS_IMPORT_MEATADATA_ARR

#ifdef WAVELET_RS_IMPORT_MEATADATA_SLICE
#define DECODER_METADATA_SLICE_EXTERN(T, F, N, M, MN) \
    DECODER_METADATA_EXTERN_(T, F, N, owned_slice<M>, MN##_slice)
#else
#define DECODER_METADATA_SLICE_EXTERN(T, F, N, M, MN)
#endif // WAVELET_RS_IMPORT_MEATADATA_SLICE

#define DECODER_METADATA_EXTERN(T, F, N, M, MN)   \
    DECODER_METADATA_EXTERN_(T, F, N, M, MN)      \
    DECODER_METADATA_ARRAY_EXTERN(T, F, N, M, MN) \
    DECODER_METADATA_SLICE_EXTERN(T, F, N, M, MN)

#define DECODER_EXTERN_(T, F, N)                                                                                                          \
    static_assert(std::is_standard_layout<reader_fetcher<T>>::value, "reader_fetcher<" #T "> must be a standard-layout type");            \
    static_assert(std::is_standard_layout<writer_fetcher<T>>::value, "writer_fetcher<" #T "> must be a standard-layout type");            \
    static_assert(std::is_standard_layout<slice<range<std::size_t>>>::value, "slice<range<std::size_t>> must be a standard-layout type"); \
    extern "C" decoder_* wavelet_rs_decoder_##N##_new(const char*);                                                                       \
    extern "C" void wavelet_rs_decoder_##N##_free(decoder_*);                                                                             \
    extern "C" void wavelet_rs_decoder_##N##_decode(decoder_*,                                                                            \
        writer_fetcher<T>, slice<range<std::size_t>>, slice<std::uint32_t>);                                                              \
    extern "C" void wavelet_rs_decoder_##N##_refine(decoder_*,                                                                            \
        reader_fetcher<T>, writer_fetcher<T>,                                                                                             \
        slice<range<std::size_t>>, slice<range<std::size_t>>,                                                                             \
        slice<std::uint32_t>, slice<std::uint32_t>);                                                                                      \
    DECODER_METADATA_EXTERN(T, F, N, std::uint8_t, u8)                                                                                    \
    DECODER_METADATA_EXTERN(T, F, N, std::uint16_t, u16)                                                                                  \
    DECODER_METADATA_EXTERN(T, F, N, std::uint32_t, u32)                                                                                  \
    DECODER_METADATA_EXTERN(T, F, N, std::uint64_t, u64)                                                                                  \
    DECODER_METADATA_EXTERN(T, F, N, std::int8_t, i8)                                                                                     \
    DECODER_METADATA_EXTERN(T, F, N, std::int16_t, i16)                                                                                   \
    DECODER_METADATA_EXTERN(T, F, N, std::int32_t, i32)                                                                                   \
    DECODER_METADATA_EXTERN(T, F, N, std::int64_t, i64)                                                                                   \
    DECODER_METADATA_EXTERN(T, F, N, float, f32)                                                                                          \
    DECODER_METADATA_EXTERN(T, F, N, double, f64)                                                                                         \
    DECODER_METADATA_EXTERN(T, F, N, string, string)                                                                                      \
    template <>                                                                                                                           \
    struct decoder_impl<T, F> {                                                                                                           \
        static constexpr bool implemented = true;                                                                                         \
        static constexpr auto new_fn = wavelet_rs_decoder_##N##_new;                                                                      \
        static constexpr auto free_fn = wavelet_rs_decoder_##N##_free;                                                                    \
        static constexpr auto decode_fn = wavelet_rs_decoder_##N##_decode;                                                                \
        static constexpr auto refine_fn = wavelet_rs_decoder_##N##_refine;                                                                \
    };

#ifdef WAVELET_RS_IMPORT_VEC
#define DECODER_VEC_EXTERN(T, F, N)           \
    DECODER_EXTERN_(array_1<T>, F, vec_1_##N) \
    DECODER_EXTERN_(array_2<T>, F, vec_2_##N) \
    DECODER_EXTERN_(array_3<T>, F, vec_3_##N) \
    DECODER_EXTERN_(array_4<T>, F, vec_4_##N)
#else
#define DECODER_VEC_EXTERN(T, F, N)
#endif // WAVELET_RS_IMPORT_VEC

#ifdef WAVELET_RS_IMPORT_MAT
#define DECODER_MATRIX_EXTERN(T, F, N)                   \
    DECODER_EXTERN_(array_1<array_1<T>>, F, mat_1x1_##N) \
    DECODER_EXTERN_(array_1<array_2<T>>, F, mat_1x2_##N) \
    DECODER_EXTERN_(array_1<array_3<T>>, F, mat_1x3_##N) \
    DECODER_EXTERN_(array_1<array_4<T>>, F, mat_1x4_##N) \
    DECODER_EXTERN_(array_2<array_1<T>>, F, mat_2x1_##N) \
    DECODER_EXTERN_(array_2<array_2<T>>, F, mat_2x2_##N) \
    DECODER_EXTERN_(array_2<array_3<T>>, F, mat_2x3_##N) \
    DECODER_EXTERN_(array_2<array_4<T>>, F, mat_2x4_##N) \
    DECODER_EXTERN_(array_3<array_1<T>>, F, mat_3x1_##N) \
    DECODER_EXTERN_(array_3<array_2<T>>, F, mat_3x2_##N) \
    DECODER_EXTERN_(array_3<array_3<T>>, F, mat_3x3_##N) \
    DECODER_EXTERN_(array_3<array_4<T>>, F, mat_3x4_##N) \
    DECODER_EXTERN_(array_4<array_1<T>>, F, mat_4x1_##N) \
    DECODER_EXTERN_(array_4<array_2<T>>, F, mat_4x2_##N) \
    DECODER_EXTERN_(array_4<array_3<T>>, F, mat_4x3_##N) \
    DECODER_EXTERN_(array_4<array_4<T>>, F, mat_4x4_##N)
#else
#define DECODER_MATRIX_EXTERN(T, F, N)
#endif // WAVELET_RS_IMPORT_MAT

#define DECODER_EXTERN(T, N)                          \
    DECODER_EXTERN_(T, HaarWavelet, N##_haar)         \
    DECODER_EXTERN_(T, AverageFilter, N##_average)    \
    DECODER_VEC_EXTERN(T, HaarWavelet, N##_haar)      \
    DECODER_VEC_EXTERN(T, AverageFilter, N##_average) \
    DECODER_MATRIX_EXTERN(T, HaarWavelet, N##_haar)   \
    DECODER_MATRIX_EXTERN(T, AverageFilter, N##_average)

    // NOLINTBEGIN(*-return-type-c-linkage)
    DECODER_EXTERN(float, f32)
    DECODER_EXTERN(double, f64)
    // NOLINTEND(*-return-type-c-linkage)
}

template <typename T, typename F>
class decoder {
    dec_priv_::decoder_* m_dec;

    using decoder_ = dec_priv_::decoder_impl<T, F>;
    static_assert(decoder_::implemented, "decoder is not implemented for the element, filter pair");

    template <typename U>
    using decoder_meta_ = dec_priv_::decoder_metadata_impl<T, F, U>;

public:
    /// Constructs a new decoder by opening the encoded binary.
    ///
    /// @param path path to the encoded binary file.
    decoder(const char* path)
        : m_dec { decoder_::new_fn(path) }
    {
    }

    decoder(const decoder& other) = delete;

    decoder(decoder&& other) noexcept
        : m_dec { std::exchange(other.m_dec, nullptr) }
    {
    }

    ~decoder()
    {
        if (this->m_dec != nullptr) {
            decoder_::free_fn(this->m_dec);
            this->m_dec = nullptr;
        }
    }

    decoder& operator=(const decoder& other) = delete;

    decoder& operator=(decoder&& other) noexcept
    {
        if (this != &other) {
            std::swap(this->m_dec, other.m_dec);
        }

        return *this;
    }

    /// Fetches an element from the decoder metadata.
    ///
    /// @tparam U type of the element.
    /// @param key key of the mapped element.
    /// @return metadata element.
    template <typename U>
    option<U> metadata_get(const char* key) const
    {
        static_assert(decoder_meta_<U>::implemented,
            "metadata_get is not implemented for the element, filter, metedata triplet");
        return decoder_meta_<U>::get_fn(key);
    }

    /// Decodes the dataset to the required detail levels.
    ///
    /// @param writer writer for writing into the requested output volume.
    /// @param roi region of interest for the decoding operation.
    /// @param levels desired detail levels.
    void decode(writer_fetcher<T> writer,
        slice<range<std::size_t>> roi,
        slice<std::uint32_t> levels) const
    {
        decoder_::decode_fn(this->m_dec,
            std::move(writer),
            std::move(roi),
            std::move(levels));
    }

    /// Refines a partially decoded dataset by the specified detail levels.
    ///
    /// @param reader reader for reading from a partially decoded input volume.
    /// @param writer writer for writing into the requested output volume.
    /// @param input_range range of the original volume contained in the input volume.
    /// @param output_range desired range of data to write back, must be a subrange of input_range.
    /// @param curr_levels detail levels of the input volume.
    /// @param refinements number of refinement levels to apply.
    void refine(reader_fetcher<T> reader,
        writer_fetcher<T> writer,
        slice<range<std::size_t>> input_range,
        slice<range<std::size_t>> output_range,
        slice<std::uint32_t> curr_levels,
        slice<std::uint32_t> refinements) const
    {
        decoder_::refine_fn(this->m_dec,
            std::move(reader),
            std::move(writer),
            std::move(input_range),
            std::move(output_range),
            std::move(curr_levels),
            std::move(refinements));
    }
};

}

#endif // !WAVELET_H