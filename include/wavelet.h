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

#ifndef WAVELET_RS_FEAT_FFI
#error "wavelet-rs must be compiled with the 'ffi' feature"
#endif // !WAVELET_FEAT_FFI

#ifdef WAVELET_RS_IMPORT_ALL
#define WAVELET_RS_IMPORT_VEC
#define WAVELET_RS_IMPORT_MAT
#define WAVELET_RS_IMPORT_MEATADATA_ARR
#define WAVELET_RS_IMPORT_MEATADATA_SLICE
#endif // WAVELET_RS_IMPORT_ALL

#if defined(WAVELET_RS_IMPORT_VEC) && !defined(WAVELET_RS_FEAT_FFI_VEC)
#error "wavelet-rs was not compiled with the required 'ffi_vec' feature (required because of 'WAVELET_RS_IMPORT_VEC')"
#endif // WAVELET_RS_IMPORT_VEC && !WAVELET_RS_FEAT_FFI_VEC

#if defined(WAVELET_RS_IMPORT_MAT) && !defined(WAVELET_RS_FEAT_FFI_MAT)
#error "wavelet-rs was not compiled with the required 'ffi_mat' feature (required because of 'WAVELET_RS_IMPORT_MAT')"
#endif // WAVELET_RS_IMPORT_MAT && !WAVELET_RS_FEAT_FFI_MAT

#if defined(WAVELET_RS_IMPORT_MEATADATA_ARR) && !defined(WAVELET_RS_FEAT_FFI_MEATADATA_ARR)
#error "wavelet-rs was not compiled with the required 'ffi_metadata_arr' feature (required because of 'WAVELET_RS_IMPORT_MEATADATA_ARR')"
#endif // WAVELET_RS_IMPORT_MEATADATA_ARR && !WAVELET_RS_FEAT_FFI_MEATADATA_ARR

#if defined(WAVELET_RS_IMPORT_MEATADATA_SLICE) && !defined(WAVELET_RS_FEAT_FFI_MEATADATA_SLICE)
#error "wavelet-rs was not compiled with the required 'ffi_metadata_slice' feature (required because of 'WAVELET_RS_IMPORT_MEATADATA_SLICE')"
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

namespace priv_ {
    template <class T, class U = T>
    T exchange(T& obj, U&& new_value)
    {
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201402L) || __cplusplus >= 201402L)
        return std::exchange(obj, std::forward<U>(new_value));
#else
        T old_value = std::move(obj);
        obj = std::forward<U>(new_value);
        return old_value;
#endif
    }

    template <typename T>
    union maybe_uninit {
        T elem;

        maybe_uninit() { }

        maybe_uninit(T&& x)
            : elem { std::forward<T>(x) }
        {
        }

        ~maybe_uninit() noexcept { }

        T&& get()
        {
            return std::move(elem);
        }
    };
}

/// Definition of a noninclusive range.
///
/// @tparam T range element type.
template <typename T>
struct range {
    T start;
    T end;
};

/// View over a borrowed contiguous memory region.
///
/// @tparam T Element type.
template <typename T>
class slice {
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

    /// Constructs a new empty slice.
    explicit slice() noexcept
        : m_ptr { reinterpret_cast<T*>(alignof(value_type)) }
        , m_len { 0 }
    {
    }

    /// Constructs a new slice containing one element.
    ///
    /// @param value value the slice points to.
    explicit slice(T& value) noexcept
        : m_ptr { &value }
        , m_len { 1 }
    {
    }

    /// Constructs a new slice from a pointer and a length.
    /// The pointer must be properly aligned and valid in the range [ptr; ptr+size].
    ///
    /// @param ptr start of the array.
    /// @param size length of the array.
    explicit slice(T* ptr, size_type size) noexcept
        : m_ptr { ptr }
        , m_len { size }
    {
    }

    /// @brief Constructs a new slice pointing to an array.
    ///
    /// @tparam N size of the array.
    /// @param array array the slice points to.
    template <std::size_t N>
    explicit slice(std::array<T, N>& array) noexcept
        : m_ptr { array.data() }
        , m_len { array.size() }
    {
    }

    slice(const slice& other) = default;
    slice(slice&& other) noexcept = default;

    slice& operator=(const slice& other) = default;
    slice& operator=(slice&& other) = default;

    operator slice<const T>() const noexcept
    {
        return slice<const T> { this->data(), this->size() };
    }

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

class string;

namespace slice_priv_ {
    struct own_slice_ {
        void* ptr;
        std::size_t len;
    };

    template <typename T>
    struct own_slice_impl {
        static constexpr bool implemented = false;
    };

#define OWNED_SLICE_EXTERN(T, N)                                      \
    extern "C" own_slice_ wavelet_rs_slice_##N##_new(slice<const T>); \
    extern "C" void wavelet_rs_slice_##N##_free(own_slice_);          \
    template <>                                                       \
    struct own_slice_impl<T> {                                        \
        static constexpr bool implemented = true;                     \
        static constexpr auto new_fn = wavelet_rs_slice_##N##_new;    \
        static constexpr auto free_fn = wavelet_rs_slice_##N##_free;  \
    };

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
    OWNED_SLICE_EXTERN(string, CString)
}

/// Owned view over a sontiguous memory region.
///
/// @tparam T Element type.
template <typename T>
class owned_slice {
    slice<T> m_slice;

    using own_slice_ = slice_priv_::own_slice_impl<T>;
    static_assert(own_slice_::implemented, "owned_slice is not implemented for the element");

public:
    typedef typename slice<T>::value_type value_type;
    typedef typename slice<T>::size_type size_type;
    typedef typename slice<T>::difference_type difference_type;
    typedef typename slice<T>::reference reference;
    typedef typename slice<T>::const_reference const_reference;
    typedef typename slice<T>::pointer pointer;
    typedef typename slice<T>::const_pointer const_pointer;
    typedef typename slice<T>::iterator iterator;
    typedef typename slice<T>::const_iterator const_iterator;
    typedef typename slice<T>::reverse_iterator reverse_iterator;
    typedef typename slice<T>::const_reverse_iterator const_reverse_iterator;

    /// Constructs a new empty slice.
    explicit owned_slice()
        : owned_slice { slice<T> {} }
    {
    }

    /// Constructs a new slice by copying all elements of the provided slice.
    ///
    /// @param value slice to be copied into the new slice.
    explicit owned_slice(slice<const T> s)
        : m_slice {}
    {
        static_assert(std::is_standard_layout<T>::value, "invalid layout");
        static_assert(std::is_standard_layout<slice<T>>::value, "invalid layout");

        auto tmp_sl = own_slice_::new_fn(std::move(s));
        this->m_slice = slice<T> { static_cast<T*>(tmp_sl.ptr), tmp_sl.len };
    }

    /// Constructs a new slice containing only one value.
    ///
    /// @param value Value contained.
    explicit owned_slice(const T& value)
        : owned_slice { slice<const T> { value } }
    {
    }

    /// Constructs a new slice from a pointer and a length.
    /// The pointer must be properly aligned and valid in the range [ptr; ptr+size].
    ///
    /// @param ptr start of the array.
    /// @param size length of the array.
    explicit owned_slice(const T* ptr, size_type size)
        : owned_slice { slice<const T> { ptr, size } }
    {
    }

    /// @brief Constructs a new slice from an array.
    ///
    /// @tparam N size of the array.
    /// @param array array the slice points to.
    template <std::size_t N>
    explicit owned_slice(const std::array<T, N>& array)
        : owned_slice {
            slice<const T> { array }
        }
    {
    }

    /// Copies another slice.
    ///
    /// @param other other slice.
    owned_slice(const owned_slice& other)
        : owned_slice { static_cast<slice<const T>>(other) }
    {
    }

    ~owned_slice()
    {
        if (this->m_slice.size() != 0) {
            own_slice_::free_fn(slice_priv_::own_slice_ { static_cast<void*>(this->data()), this->size() });
            this->m_slice = slice<T> {};
        }
    }

    owned_slice(owned_slice&& other) noexcept
        : m_slice { priv_::exchange(other.m_slice, slice<T> {}) }
    {
    }

    owned_slice& operator=(const owned_slice& rhs)
    {
        if (this != &rhs) {
            if (this->m_slice.size() == rhs.m_slice.size()) {
                std::copy(rhs.begin(), rhs.end(), this->begin());
            } else {
                this->~owned_slice();
                new (this) owned_slice { static_cast<slice<const T>>(rhs) };
            }
        }

        return *this;
    }

    owned_slice& operator=(owned_slice&& rhs) noexcept
    {
        if (this != &rhs) {
            std::swap(this->m_slice, rhs->m_slice);
        }

        return *this;
    }

    explicit operator slice<T>() noexcept
    {
        return this->m_slice;
    }

    explicit operator slice<const T>() const noexcept
    {
        return this->m_slice;
    }

    /// Checks that the elements of the two slices are equal.
    /// Is equivalent to calling std::equal with the two slices.
    ///
    /// @tparam U other element type.
    /// @param other other slice.
    /// @return `true` if the two slices are equal, `false` otherwise.
    template <typename U>
    bool operator==(const owned_slice<U>& other) const
    {
        return this->m_slice == other.m_slice;
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
        return this->m_slice != other.m_slice;
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
        return this->m_slice < other.m_slice;
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
        return this->m_slice <= other.m_slice;
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
        return this->m_slice > other.m_slice;
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
        return this->m_slice >= other.m_slice;
    }

    /// Returns a reference to the element at specified location pos, with bounds checking.
    /// If pos is not within the range of the container, an exception of type std::out_of_range
    /// is thrown.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    reference at(size_type pos)
    {
        return this->m_slice.at(pos);
    }

    /// Returns a reference to the element at specified location pos, with bounds checking.
    /// If pos is not within the range of the container, an exception of type std::out_of_range
    /// is thrown.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    const_reference at(size_type pos) const
    {
        return this->m_slice.at(pos);
    }

    /// Returns a reference to the element at specified location pos.
    /// No bounds checking is performed.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    reference operator[](size_type pos)
    {
        return this->m_slice[pos];
    }

    /// Returns a reference to the element at specified location pos.
    /// No bounds checking is performed.
    ///
    /// @param pos position of the element to return.
    /// @return Reference to the requested element.
    const_reference operator[](size_type pos) const
    {
        return this->m_slice[pos];
    }

    /// Returns a reference to the first element in the container.
    /// @return Calling front on an empty container is undefined.
    reference front()
    {
        return this->m_slice.front();
    }

    /// Returns a reference to the first element in the container.
    /// @return Calling front on an empty container is undefined.
    const_reference front() const
    {
        return this->m_slice.front();
    }

    /// Returns a reference to the last element in the container.
    /// @return Calling back on an empty container is undefined.
    reference back()
    {
        return this->m_slice.back();
    }

    /// Returns a reference to the last element in the container.
    /// @return Calling back on an empty container is undefined.
    const_reference back() const
    {
        return this->m_slice.back();
    }

    /// Returns pointer to the underlying array serving as element storage.
    /// @return Pointer to the underlying element storage.
    T* data() noexcept
    {
        return this->m_slice.data();
    }

    /// Returns pointer to the underlying array serving as element storage.
    /// @return Pointer to the underlying element storage.
    const T* data() const noexcept
    {
        return this->m_slice.data();
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    iterator begin() noexcept
    {
        return this->m_slice.begin();
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    const_iterator begin() const noexcept
    {
        return this->m_slice.begin();
    }

    /// Returns an iterator to the first element of the array.
    /// If the array is empty, the returned iterator will be equal to end().
    ///
    /// @return Iterator to the first element.
    const_iterator cbegin() const noexcept
    {
        return this->m_slice.cbegin();
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    iterator end() noexcept
    {
        return this->m_slice.end();
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    const_iterator end() const noexcept
    {
        return this->m_slice.end();
    }

    /// Returns an iterator past the last element of the array.
    /// This element acts as a placeholder; attempting to access it results in undefined behavior.
    ///
    /// @return Iterator to the element following the last element.
    const_iterator cend() const noexcept
    {
        return this->m_slice.cend();
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    reverse_iterator rbegin() noexcept
    {
        return this->m_slice.rbegin();
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    const_reverse_iterator rbegin() const noexcept
    {
        return this->m_slice.rbegin();
    }

    /// Returns a reverse iterator to the first element of the reversed array.
    /// It corresponds to the last element of the non-reversed array. If the array is empty,
    /// the returned iterator is equal to rend().
    ///
    /// @return Reverse iterator to the first element.
    const_reverse_iterator crbegin() const noexcept
    {
        return this->m_slice.crbegin();
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    reverse_iterator rend() noexcept
    {
        return this->m_slice.rend();
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    const_reverse_iterator rend() const noexcept
    {
        return this->m_slice.rend();
    }

    /// Returns a reverse iterator to the element following the last element of the reversed array.
    /// It corresponds to the element preceding the first element of the non-reversed array.
    ///
    /// @return Reverse iterator to the element following the last element.
    const_reverse_iterator crend() const noexcept
    {
        return this->m_slice.crend();
    }

    /// Checks if the container has no elements, i.e. whether `begin() == end()`.
    ///
    /// @return `true` if the container is empty, `false` otherwise.
    [[nodiscard]] bool empty() const noexcept
    {
        return this->m_slice.empty();
    }

    /// Returns the number of elements in the container, i.e. `std::distance(begin(), end())`.
    ///
    /// @return The number of elements in the container.
    [[nodiscard]] size_type size() const noexcept
    {
        return this->m_slice.size();
    }

    /// Assigns the `value` to all elements in the container.
    ///
    /// @param value the value to assign to the elements
    void fill(const T& value)
    {
        this->m_slice.fill(value);
    }
};

/// Owned string type.
class string {
    owned_slice<char> m_buff;

public:
    typedef char CharT;
    typedef typename owned_slice<char>::value_type value_type;
    typedef typename owned_slice<char>::size_type size_type;
    typedef typename owned_slice<char>::difference_type difference_type;
    typedef typename owned_slice<char>::reference reference;
    typedef typename owned_slice<char>::const_reference const_reference;
    typedef typename owned_slice<char>::pointer pointer;
    typedef typename owned_slice<char>::const_pointer const_pointer;
    typedef typename owned_slice<char>::iterator iterator;
    typedef typename owned_slice<char>::const_iterator const_iterator;
    typedef typename owned_slice<char>::reverse_iterator reverse_iterator;
    typedef typename owned_slice<char>::const_reverse_iterator const_reverse_iterator;

    explicit string() = default;
    explicit string(const CharT* s, size_type count)
        : m_buff { slice<const char> { s, count } }
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

/// Element types exposed by the ffi api.
enum class elem_type : std::int32_t {
    F32 = 0,

#ifdef WAVELET_RS_IMPORT_VEC
    F32Vec1 = 1,
    F32Vec2,
    F32Vec3,
    F32Vec4,
#endif // WAVELET_RS_IMPORT_VEC

#ifdef WAVELET_RS_IMPORT_MAT
    F32Mat1x1 = 11,
    F32Mat1x2,
    F32Mat1x3,
    F32Mat1x4,

    F32Mat2x1,
    F32Mat2x2,
    F32Mat2x3,
    F32Mat2x4,

    F32Mat3x1,
    F32Mat3x2,
    F32Mat3x3,
    F32Mat3x4,

    F32Mat4x1,
    F32Mat4x2,
    F32Mat4x3,
    F32Mat4x4,
#endif // WAVELET_RS_IMPORT_MAT

    F64 = 40,

#ifdef WAVELET_RS_IMPORT_VEC
    F64Vec1 = 41,
    F64Vec2,
    F64Vec3,
    F64Vec4,
#endif // WAVELET_RS_IMPORT_VEC

#ifdef WAVELET_RS_IMPORT_MAT
    F64Mat1x1 = 51,
    F64Mat1x2,
    F64Mat1x3,
    F64Mat1x4,

    F64Mat2x1,
    F64Mat2x2,
    F64Mat2x3,
    F64Mat2x4,

    F64Mat3x1,
    F64Mat3x2,
    F64Mat3x3,
    F64Mat3x4,

    F64Mat4x1,
    F64Mat4x2,
    F64Mat4x3,
    F64Mat4x4,
#endif // WAVELET_RS_IMPORT_MAT
};

/// Info pertaining to a decoder.
struct decoder_info {
    elem_type e_type;
    owned_slice<std::size_t> dims;
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
        this->m_call = +[](ctx_t ctx, Args... args) -> Ret {
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
        this->m_call = +[](ctx_t ctx, Args... args) -> Ret {
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
        this->m_call = +[](ctx_t ctx, Args... args) -> Ret {
            const F_* ptr = static_cast<const F_*>(ctx.allocated);
            return (*ptr)(std::move(args)...);
        };
    }

    callable(const callable& other) = delete;

    callable(callable&& other) noexcept
        : m_ctx { other.m_ctx }
        , m_drop { priv_::exchange(other.m_drop, empty_drop) }
        , m_call { priv_::exchange(other.m_call, empty_call) }
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
using volume_fetcher = callable<Fn, T, slice<const std::size_t>>;

/// Callable reading an element from the provided position in the current block.
///
/// @tparam T element type of the volume.
template <typename T>
using block_reader = callable<Fn, T, slice<const std::size_t>>;

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
using reader_fetcher = callable<FnOnce, block_reader_fetcher<T>, slice<const std::size_t>>;

/// Callable writing an element at the provided position in the current block.
///
/// @tparam T element type of the volume.
template <typename T>
using block_writer = callable<FnMut, void, slice<const std::size_t>, T>;

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
using writer_fetcher = callable<FnOnce, block_writer_fetcher<T>, slice<const std::size_t>>;

namespace filters {
    /// Marker type for the haar wavelet.
    struct haar_wavelet;

    /// Marker type for the average filter.
    struct average_filter;
}

///////////////// Encoder /////////////////

template <typename T>
using array_1 = std::array<T, 1>;
template <typename T>
using array_2 = std::array<T, 2>;
template <typename T>
using array_3 = std::array<T, 3>;
template <typename T>
using array_4 = std::array<T, 4>;

namespace enc_priv_ {
    class encoder_;

    template <typename T>
    struct encoder_impl {
        static constexpr bool implemented = false;
    };

    template <typename T, typename F>
    struct encoder_enc_impl {
        static constexpr bool implemented = false;
    };

    template <typename T, typename U>
    struct encoder_metadata_impl {
        static constexpr bool implemented = false;
    };

#define ENCODER_METADATA_EXTERN_(T, N, M, MN)                                                                                  \
    extern "C" void wavelet_rs_encoder_##N##_metadata_get_##MN(const encoder_*, const char*, priv_::maybe_uninit<option<M>>*); \
    extern "C" std::uint8_t wavelet_rs_encoder_##N##_metadata_insert_##MN(encoder_*, const char*, priv_::maybe_uninit<M>*);    \
    template <>                                                                                                                \
    struct encoder_metadata_impl<T, M> {                                                                                       \
        static constexpr bool implemented = true;                                                                              \
        static constexpr auto get_fn = wavelet_rs_encoder_##N##_metadata_get_##MN;                                             \
        static constexpr auto insert_fn = wavelet_rs_encoder_##N##_metadata_insert_##MN;                                       \
    };

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

#define ENCODER_EXTERN_(T, N)                                                        \
    extern "C" encoder_* wavelet_rs_encoder_##N##_new(                               \
        const slice<const std::size_t>*, std::size_t);                               \
    extern "C" void wavelet_rs_encoder_##N##_free(encoder_*);                        \
    extern "C" void wavelet_rs_encoder_##N##_add_fetcher(encoder_*,                  \
        const slice<const std::size_t>*,                                             \
        const priv_::maybe_uninit<volume_fetcher<T>>*);                              \
    extern "C" void wavelet_rs_encoder_##N##_encode_haar(const encoder_*,            \
        const char*, const slice<const std::size_t>*);                               \
    extern "C" void wavelet_rs_encoder_##N##_encode_average(const encoder_*,         \
        const char*, const slice<const std::size_t>*);                               \
    ENCODER_METADATA_EXTERN(T, N, std::uint8_t, u8)                                  \
    ENCODER_METADATA_EXTERN(T, N, std::uint16_t, u16)                                \
    ENCODER_METADATA_EXTERN(T, N, std::uint32_t, u32)                                \
    ENCODER_METADATA_EXTERN(T, N, std::uint64_t, u64)                                \
    ENCODER_METADATA_EXTERN(T, N, std::int8_t, i8)                                   \
    ENCODER_METADATA_EXTERN(T, N, std::int16_t, i16)                                 \
    ENCODER_METADATA_EXTERN(T, N, std::int32_t, i32)                                 \
    ENCODER_METADATA_EXTERN(T, N, std::int64_t, i64)                                 \
    ENCODER_METADATA_EXTERN(T, N, float, f32)                                        \
    ENCODER_METADATA_EXTERN(T, N, double, f64)                                       \
    ENCODER_METADATA_EXTERN_(T, N, string, string)                                   \
    template <>                                                                      \
    struct encoder_impl<T> {                                                         \
        static constexpr bool implemented = true;                                    \
        static constexpr auto new_fn = wavelet_rs_encoder_##N##_new;                 \
        static constexpr auto free_fn = wavelet_rs_encoder_##N##_free;               \
        static constexpr auto add_fetcher_fn = wavelet_rs_encoder_##N##_add_fetcher; \
    };                                                                               \
    template <>                                                                      \
    struct encoder_enc_impl<T, filters::haar_wavelet> {                              \
        static constexpr bool implemented = true;                                    \
        static constexpr auto encode_fn = wavelet_rs_encoder_##N##_encode_haar;      \
    };                                                                               \
    template <>                                                                      \
    struct encoder_enc_impl<T, filters::average_filter> {                            \
        static constexpr bool implemented = true;                                    \
        static constexpr auto encode_fn = wavelet_rs_encoder_##N##_encode_average;   \
    };

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

    ENCODER_EXTERN(float, f32)
    ENCODER_EXTERN(double, f64)
}

template <typename T>
class encoder {
    enc_priv_::encoder_* m_enc;

    using encoder_ = enc_priv_::encoder_impl<T>;
    static_assert(encoder_::implemented, "encoder is not implemented for the element");

    template <typename F>
    using encoder_enc_ = enc_priv_::encoder_enc_impl<T, F>;

    template <typename U>
    using encoder_meta_ = enc_priv_::encoder_metadata_impl<T, U>;

public:
    /// Constructs a new encoder spanning a volume with dimmensions `dims`.
    ///
    /// @param dims dimmensions of the volume to be encoded.
    /// @param num_base_dims number of dimmensions contained in each volume_fetcher.
    encoder(slice<const std::size_t> dims, std::size_t num_base_dims)
        : m_enc { encoder_::new_fn(&dims, num_base_dims) }
    {
    }

    encoder(const encoder& other) = delete;

    encoder(encoder&& other) noexcept
        : m_enc { priv_::exchange(other.m_enc, nullptr) }
    {
    }

    ~encoder()
    {
        if (this->m_enc != nullptr) {
            encoder_::free_fn(this->m_enc);
            this->m_enc = nullptr;
        }
    }

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
    void add_fetcher(slice<const std::size_t> index, volume_fetcher<T>&& fetcher)
    {
        static_assert(std::is_standard_layout<slice<const std::size_t>>::value, "invalid layout");
        static_assert(std::is_standard_layout<priv_::maybe_uninit<volume_fetcher<T>>>::value, "invalid layout");

        priv_::maybe_uninit<volume_fetcher<T>> f { std::move(fetcher) };
        encoder_::add_fetcher_fn(this->m_enc, &index, &f);
    }

    /// Encodes the volume using the inserted fetchers and the specified filter.
    /// The output will be written into the `output` directory. The caller must
    /// ensure that `output` is a writable directory. The selected block size
    /// must tile the input volume exactly.
    ///
    /// @tparam Filter filter to use for the encoding.
    /// @param output output directory.
    /// @param block_size selected block size.
    template <typename Filter>
    void encode(const char* output, slice<const std::size_t> block_size) const
    {
        static_assert(std::is_standard_layout<slice<const std::size_t>>::value, "invalid layout");
        static_assert(encoder_enc_<Filter>::implemented,
            "encode is not implemented for the element, filter pair");

        encoder_enc_<Filter>::encode_fn(this->m_enc, output, &block_size);
    }

    /// Fetches an element from the encoder metadata.
    ///
    /// @tparam U type of the element.
    /// @param key key of the mapped element.
    /// @return metadata element.
    template <typename U>
    option<U> metadata_get(const char* key) const
    {
        static_assert(std::is_standard_layout<priv_::maybe_uninit<option<U>>>::value, "invalid layout");
        static_assert(encoder_meta_<U>::implemented,
            "metadata_get is not implemented for the element, metadata pair");

        priv_::maybe_uninit<option<U>> res {};
        encoder_meta_<U>::get_fn(this->m_enc, key, &res);
        return res.get();
    }

    /// Inserts an element into the encoder metadata.
    ///
    /// @tparam U type of the element.
    /// @param key key where the element will be mapped to.
    /// @param value element to insert into the metadata.
    /// @return `true` if an old value was overwritten, `false` otherwise.
    template <typename U>
    bool metadata_insert(const char* key, U&& value)
    {
        static_assert(std::is_standard_layout<priv_::maybe_uninit<U>>::value, "invalid layout");
        static_assert(encoder_meta_<U>::implemented,
            "metadata_insert is not implemented for the element, metadata pair");

        priv_::maybe_uninit<U> v { std::forward<U>(value) };
        return encoder_meta_<U>::insert_fn(this->m_enc, key, &v);
    }
};

///////////////// Decoder /////////////////

namespace dec_priv_ {
    class decoder_;

    template <typename T>
    struct decoder_impl {
        static constexpr bool implemented = false;
    };

    template <typename T, typename U>
    struct decoder_metadata_impl {
        static constexpr bool implemented = false;
    };

#define DECODER_METADATA_EXTERN_(T, N, M, MN)                                                                                  \
    extern "C" void wavelet_rs_decoder_##N##_metadata_get_##MN(const decoder_*, const char*, priv_::maybe_uninit<option<M>>*); \
    template <>                                                                                                                \
    struct decoder_metadata_impl<T, M> {                                                                                       \
        static constexpr bool implemented = true;                                                                              \
        static constexpr auto get_fn = wavelet_rs_decoder_##N##_metadata_get_##MN;                                             \
    };

#ifdef WAVELET_RS_IMPORT_MEATADATA_ARR
#define DECODER_METADATA_ARRAY_EXTERN(T, N, M, MN)         \
    DECODER_METADATA_EXTERN_(T, N, array_1<M>, MN##_arr_1) \
    DECODER_METADATA_EXTERN_(T, N, array_2<M>, MN##_arr_2) \
    DECODER_METADATA_EXTERN_(T, N, array_3<M>, MN##_arr_3) \
    DECODER_METADATA_EXTERN_(T, N, array_4<M>, MN##_arr_4)
#else
#define DECODER_METADATA_ARRAY_EXTERN(T, N, M, MN)
#endif // WAVELET_RS_IMPORT_MEATADATA_ARR

#ifdef WAVELET_RS_IMPORT_MEATADATA_SLICE
#define DECODER_METADATA_SLICE_EXTERN(T, N, M, MN) \
    DECODER_METADATA_EXTERN_(T, N, owned_slice<M>, MN##_slice)
#else
#define DECODER_METADATA_SLICE_EXTERN(T, N, M, MN)
#endif // WAVELET_RS_IMPORT_MEATADATA_SLICE

#define DECODER_METADATA_EXTERN(T, N, M, MN)   \
    DECODER_METADATA_EXTERN_(T, N, M, MN)      \
    DECODER_METADATA_ARRAY_EXTERN(T, N, M, MN) \
    DECODER_METADATA_SLICE_EXTERN(T, N, M, MN)

#define DECODER_EXTERN_(T, N)                                              \
    extern "C" decoder_* wavelet_rs_decoder_##N##_new(const char*);        \
    extern "C" void wavelet_rs_decoder_##N##_free(decoder_*);              \
    extern "C" void wavelet_rs_decoder_##N##_dims(const decoder_*,         \
        priv_::maybe_uninit<slice<const std::size_t>>*);                   \
    extern "C" void wavelet_rs_decoder_##N##_decode(const decoder_*,       \
        const priv_::maybe_uninit<writer_fetcher<T>>*,                     \
        const slice<const range<std::size_t>>*,                            \
        const slice<const std::uint32_t>*);                                \
    extern "C" void wavelet_rs_decoder_##N##_refine(const decoder_*,       \
        const priv_::maybe_uninit<reader_fetcher<T>>*,                     \
        const priv_::maybe_uninit<writer_fetcher<T>>*,                     \
        const slice<const range<std::size_t>>*,                            \
        const slice<const range<std::size_t>>*,                            \
        const slice<const std::uint32_t>*,                                 \
        const slice<const std::uint32_t>*);                                \
    DECODER_METADATA_EXTERN(T, N, std::uint8_t, u8)                        \
    DECODER_METADATA_EXTERN(T, N, std::uint16_t, u16)                      \
    DECODER_METADATA_EXTERN(T, N, std::uint32_t, u32)                      \
    DECODER_METADATA_EXTERN(T, N, std::uint64_t, u64)                      \
    DECODER_METADATA_EXTERN(T, N, std::int8_t, i8)                         \
    DECODER_METADATA_EXTERN(T, N, std::int16_t, i16)                       \
    DECODER_METADATA_EXTERN(T, N, std::int32_t, i32)                       \
    DECODER_METADATA_EXTERN(T, N, std::int64_t, i64)                       \
    DECODER_METADATA_EXTERN(T, N, float, f32)                              \
    DECODER_METADATA_EXTERN(T, N, double, f64)                             \
    DECODER_METADATA_EXTERN(T, N, string, string)                          \
    template <>                                                            \
    struct decoder_impl<T> {                                               \
        static constexpr bool implemented = true;                          \
        static constexpr auto new_fn = wavelet_rs_decoder_##N##_new;       \
        static constexpr auto free_fn = wavelet_rs_decoder_##N##_free;     \
        static constexpr auto dims_fn = wavelet_rs_decoder_##N##_dims;     \
        static constexpr auto decode_fn = wavelet_rs_decoder_##N##_decode; \
        static constexpr auto refine_fn = wavelet_rs_decoder_##N##_refine; \
    };

#ifdef WAVELET_RS_IMPORT_VEC
#define DECODER_VEC_EXTERN(T, N)           \
    DECODER_EXTERN_(array_1<T>, vec_1_##N) \
    DECODER_EXTERN_(array_2<T>, vec_2_##N) \
    DECODER_EXTERN_(array_3<T>, vec_3_##N) \
    DECODER_EXTERN_(array_4<T>, vec_4_##N)
#else
#define DECODER_VEC_EXTERN(T, N)
#endif // WAVELET_RS_IMPORT_VEC

#ifdef WAVELET_RS_IMPORT_MAT
#define DECODER_MATRIX_EXTERN(T, N)                   \
    DECODER_EXTERN_(array_1<array_1<T>>, mat_1x1_##N) \
    DECODER_EXTERN_(array_1<array_2<T>>, mat_1x2_##N) \
    DECODER_EXTERN_(array_1<array_3<T>>, mat_1x3_##N) \
    DECODER_EXTERN_(array_1<array_4<T>>, mat_1x4_##N) \
    DECODER_EXTERN_(array_2<array_1<T>>, mat_2x1_##N) \
    DECODER_EXTERN_(array_2<array_2<T>>, mat_2x2_##N) \
    DECODER_EXTERN_(array_2<array_3<T>>, mat_2x3_##N) \
    DECODER_EXTERN_(array_2<array_4<T>>, mat_2x4_##N) \
    DECODER_EXTERN_(array_3<array_1<T>>, mat_3x1_##N) \
    DECODER_EXTERN_(array_3<array_2<T>>, mat_3x2_##N) \
    DECODER_EXTERN_(array_3<array_3<T>>, mat_3x3_##N) \
    DECODER_EXTERN_(array_3<array_4<T>>, mat_3x4_##N) \
    DECODER_EXTERN_(array_4<array_1<T>>, mat_4x1_##N) \
    DECODER_EXTERN_(array_4<array_2<T>>, mat_4x2_##N) \
    DECODER_EXTERN_(array_4<array_3<T>>, mat_4x3_##N) \
    DECODER_EXTERN_(array_4<array_4<T>>, mat_4x4_##N)
#else
#define DECODER_MATRIX_EXTERN(T, N)
#endif // WAVELET_RS_IMPORT_MAT

#define DECODER_EXTERN(T, N) \
    DECODER_EXTERN_(T, N)    \
    DECODER_VEC_EXTERN(T, N) \
    DECODER_MATRIX_EXTERN(T, N)

    DECODER_EXTERN(float, f32)
    DECODER_EXTERN(double, f64)

    extern "C" void get_decoder_info(const char*, priv_::maybe_uninit<decoder_info>*);
}

template <typename T>
class decoder_ref {
    dec_priv_::decoder_* m_dec;

    using decoder_ = dec_priv_::decoder_impl<T>;
    static_assert(decoder_::implemented, "decoder is not implemented for the element type");

    template <typename U>
    using decoder_meta_ = dec_priv_::decoder_metadata_impl<T, U>;

public:
    /// Constructs a new decoder reference.
    decoder_ref(dec_priv_::decoder_* dec) noexcept
        : m_dec { dec }
    {
    }

    decoder_ref(const decoder_ref& other) noexcept = default;
    decoder_ref(decoder_ref&& other) noexcept = default;

    decoder_ref& operator=(const decoder_ref& other) = default;
    decoder_ref& operator=(decoder_ref&& other) noexcept = default;

    /// Fetches the dimensions of the encoded dataset.
    ///
    /// @return Dimensions of the encoded dataset.
    slice<const std::size_t> dims() const
    {
        static_assert(std::is_standard_layout<priv_::maybe_uninit<slice<const std::size_t>>>::value, "invalid layout");

        priv_::maybe_uninit<slice<const std::size_t>> res {};
        decoder_::dims_fn(this->m_dec, &res);
        return res.get();
    }

    /// Fetches an element from the decoder metadata.
    ///
    /// @tparam U type of the element.
    /// @param key key of the mapped element.
    /// @return metadata element.
    template <typename U>
    option<U> metadata_get(const char* key) const
    {
        static_assert(std::is_standard_layout<priv_::maybe_uninit<option<U>>>::value, "invalid layout");
        static_assert(decoder_meta_<U>::implemented,
            "metadata_get is not implemented for the element, metadata pair");

        priv_::maybe_uninit<option<U>> res {};
        decoder_meta_<U>::get_fn(this->m_dec, key, &res);
        return res.get();
    }

    /// Decodes the dataset to the required detail levels.
    ///
    /// @param writer writer for writing into the requested output volume.
    /// @param roi region of interest for the decoding operation.
    /// @param levels desired detail levels.
    void decode(writer_fetcher<T>&& writer,
        slice<const range<std::size_t>> roi,
        slice<const std::uint32_t> levels) const
    {
        static_assert(std::is_standard_layout<priv_::maybe_uninit<writer_fetcher<T>>>::value, "invalid layout");
        static_assert(std::is_standard_layout<slice<const range<std::size_t>>>::value, "invalid layout");
        static_assert(std::is_standard_layout<slice<const std::uint32_t>>::value, "invalid layout");

        priv_::maybe_uninit<writer_fetcher<T>> w { std::forward<writer_fetcher<T>>(writer) };
        decoder_::decode_fn(this->m_dec, &w, &roi, &levels);
    }

    /// Refines a partially decoded dataset by the specified detail levels.
    ///
    /// @param reader reader for reading from a partially decoded input volume.
    /// @param writer writer for writing into the requested output volume.
    /// @param input_range range of the original volume contained in the input volume.
    /// @param output_range desired range of data to write back, must be a subrange of input_range.
    /// @param curr_levels detail levels of the input volume.
    /// @param refinements number of refinement levels to apply.
    void refine(reader_fetcher<T>&& reader,
        writer_fetcher<T>&& writer,
        slice<const range<std::size_t>> input_range,
        slice<const range<std::size_t>> output_range,
        slice<const std::uint32_t> curr_levels,
        slice<const std::uint32_t> refinements) const
    {
        static_assert(std::is_standard_layout<priv_::maybe_uninit<reader_fetcher<T>>>::value, "invalid layout");
        static_assert(std::is_standard_layout<priv_::maybe_uninit<writer_fetcher<T>>>::value, "invalid layout");
        static_assert(std::is_standard_layout<slice<const range<std::size_t>>>::value, "invalid layout");
        static_assert(std::is_standard_layout<slice<const std::uint32_t>>::value, "invalid layout");

        priv_::maybe_uninit<reader_fetcher<T>> r { std::forward<reader_fetcher<T>>(reader) };
        priv_::maybe_uninit<writer_fetcher<T>> w { std::forward<writer_fetcher<T>>(writer) };
        decoder_::refine_fn(this->m_dec, &reader, &writer, &input_range,
            &output_range, &curr_levels, &refinements);
    }

    /// Returns the pointer to the type-erased decoder.
    ///
    /// @return pointer to the decoder
    dec_priv_::decoder_* pointer() const noexcept
    {
        return this->m_dec;
    }
};

template <typename T>
class decoder {
    decoder_ref<T> m_dec;

    using decoder_ = dec_priv_::decoder_impl<T>;
    static_assert(decoder_::implemented, "decoder is not implemented for the element type");

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
        : m_dec { priv_::exchange(other.m_dec, nullptr) }
    {
    }

    ~decoder()
    {
        this->reset(decoder_ref<T> { nullptr });
    }

    decoder& operator=(const decoder& other) = delete;

    decoder& operator=(decoder&& other) noexcept
    {
        if (this != &other) {
            std::swap(this->m_dec, other.m_dec);
        }

        return *this;
    }

    /// Fetches the dimensions of the encoded dataset.
    ///
    /// @return Dimensions of the encoded dataset.
    slice<const std::size_t> dims() const
    {
        return m_dec.dims();
    }

    /// Fetches an element from the decoder metadata.
    ///
    /// @tparam U type of the element.
    /// @param key key of the mapped element.
    /// @return metadata element.
    template <typename U>
    option<U> metadata_get(const char* key) const
    {
        return m_dec.template metadata_get<U>(key);
    }

    /// Decodes the dataset to the required detail levels.
    ///
    /// @param writer writer for writing into the requested output volume.
    /// @param roi region of interest for the decoding operation.
    /// @param levels desired detail levels.
    void decode(writer_fetcher<T>&& writer,
        slice<const range<std::size_t>> roi,
        slice<const std::uint32_t> levels) const
    {
        return m_dec.decode(std::move(writer), std::move(roi), std::move(levels));
    }

    /// Refines a partially decoded dataset by the specified detail levels.
    ///
    /// @param reader reader for reading from a partially decoded input volume.
    /// @param writer writer for writing into the requested output volume.
    /// @param input_range range of the original volume contained in the input volume.
    /// @param output_range desired range of data to write back, must be a subrange of input_range.
    /// @param curr_levels detail levels of the input volume.
    /// @param refinements number of refinement levels to apply.
    void refine(reader_fetcher<T>&& reader,
        writer_fetcher<T>&& writer,
        slice<const range<std::size_t>> input_range,
        slice<const range<std::size_t>> output_range,
        slice<const std::uint32_t> curr_levels,
        slice<const std::uint32_t> refinements) const
    {
        return m_dec.refine(std::move(reader), std::move(writer),
            std::move(input_range), std::move(output_range),
            std::move(curr_levels), std::move(refinements));
    }

    /// Releases ownership of the decoder.
    ///
    /// @return released decoder.
    decoder_ref<T> release() noexcept
    {
        return priv_::exchange(this->m_dec, decoder_ref<T> { nullptr });
    }

    /// Replaces the owned decoder with a new one.
    ///
    /// @param dec new decoder.
    void reset(decoder_ref<T> dec)
    {
        if (this->m_dec.pointer() != nullptr) {
            decoder_::free_fn(this->m_dec.pointer());
            this->m_dec = dec;
        }
    }
};

/// Returns some info pertaining to the encoded data located at `path`.
inline decoder_info get_decoder_info(const char* path)
{
    static_assert(std::is_standard_layout<priv_::maybe_uninit<decoder_info>>::value, "invalid layout");

    priv_::maybe_uninit<decoder_info> res {};
    dec_priv_::get_decoder_info(path, &res);
    return res.get();
}

template <typename T>
struct elem_type_trait {
};

#define ELEM_TYPE_TRAIT_(T, N)               \
    template <>                              \
    struct elem_type_trait<T> {              \
        static constexpr elem_type type = N; \
    };

#ifdef WAVELET_RS_IMPORT_VEC
#define ELEM_TYPE_TRAIT_VEC(T, F)         \
    ELEM_TYPE_TRAIT_(array_1<T>, F##Vec1) \
    ELEM_TYPE_TRAIT_(array_2<T>, F##Vec2) \
    ELEM_TYPE_TRAIT_(array_3<T>, F##Vec3) \
    ELEM_TYPE_TRAIT_(array_4<T>, F##Vec4)
#else
#define ELEM_TYPE_TRAIT_VEC(T, F)
#endif // WAVELET_RS_IMPORT_VEC

#ifdef WAVELET_RS_IMPORT_MAT
#define ELEM_TYPE_TRAIT_MAT(T, F)                    \
    ELEM_TYPE_TRAIT_(array_1<array_1<T>>, F##Mat1x1) \
    ELEM_TYPE_TRAIT_(array_1<array_2<T>>, F##Mat1x2) \
    ELEM_TYPE_TRAIT_(array_1<array_3<T>>, F##Mat1x3) \
    ELEM_TYPE_TRAIT_(array_1<array_4<T>>, F##Mat1x4) \
    ELEM_TYPE_TRAIT_(array_2<array_1<T>>, F##Mat2x1) \
    ELEM_TYPE_TRAIT_(array_2<array_2<T>>, F##Mat2x2) \
    ELEM_TYPE_TRAIT_(array_2<array_3<T>>, F##Mat2x3) \
    ELEM_TYPE_TRAIT_(array_2<array_4<T>>, F##Mat2x4) \
    ELEM_TYPE_TRAIT_(array_3<array_1<T>>, F##Mat3x1) \
    ELEM_TYPE_TRAIT_(array_3<array_2<T>>, F##Mat3x2) \
    ELEM_TYPE_TRAIT_(array_3<array_3<T>>, F##Mat3x3) \
    ELEM_TYPE_TRAIT_(array_3<array_4<T>>, F##Mat3x4) \
    ELEM_TYPE_TRAIT_(array_4<array_1<T>>, F##Mat4x1) \
    ELEM_TYPE_TRAIT_(array_4<array_2<T>>, F##Mat4x2) \
    ELEM_TYPE_TRAIT_(array_4<array_3<T>>, F##Mat4x3) \
    ELEM_TYPE_TRAIT_(array_4<array_4<T>>, F##Mat4x4)
#else
#define ELEM_TYPE_TRAIT_MAT(T, F)
#endif // WAVELET_RS_IMPORT_MAT

#define ELEM_TYPE_TRAIT(T, N) \
    ELEM_TYPE_TRAIT_(T, N)    \
    ELEM_TYPE_TRAIT_VEC(T, N) \
    ELEM_TYPE_TRAIT_MAT(T, N)

ELEM_TYPE_TRAIT(float, elem_type::F32)
ELEM_TYPE_TRAIT(double, elem_type::F64)

};

#endif // !WAVELET_H