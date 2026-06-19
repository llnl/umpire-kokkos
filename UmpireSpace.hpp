#ifndef KOKKOS_UMPIRE_SPACE_HPP
#define KOKKOS_UMPIRE_SPACE_HPP

#include "umpire/ResourceManager.hpp"
#include <Kokkos_Core.hpp>
#include <string>
#include <cassert>
#include <concepts>
#include <optional>

template <typename MemorySpace, typename TagType = void> class UmpireSpace {
public:
  using memory_space = UmpireSpace<MemorySpace, TagType>;
  using size_type = typename MemorySpace::size_type;
  using umpire_space = UmpireSpace<MemorySpace>;

  using execution_space = typename MemorySpace::execution_space;

  using device_type = Kokkos::Device<execution_space, MemorySpace>;

  static void set_allocator(const std::string &allocator_name) {
    static int count = [allocator_name]() {
      auto &rm = umpire::ResourceManager::getInstance();
      m_allocator = rm.getAllocator(allocator_name);
      return 0;
    }();
    assert(++count == 1);
  }

  void *allocate(size_t size) const {
    assert(m_allocator.has_value());
    return m_allocator->allocate(size);
  }

private:
  template <typename ExecutionSpace>
  void *impl_allocate(const ExecutionSpace &exec, const char *arg_label,
		      const size_t arg_alloc_size,
		      const size_t arg_logical_size,
		      const Kokkos::Tools::SpaceHandle arg_handle =
                          Kokkos::Tools::make_space_handle(name())) const {
    void *ptr = allocate(arg_alloc_size);
    exec.fence(std::string("UmpireSpace<") + MemorySpace::name() + ">, fence after allocating");

    if (Kokkos::Profiling::profileLibraryLoaded()) {
      const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
      Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
    }
    return ptr;
  }

public:
  template <typename ExecutionSpace>
  void *allocate(const ExecutionSpace &exec, const char *arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return impl_allocate(exec, arg_label, arg_alloc_size, arg_logical_size);
  }

  void *allocate(const char *arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return allocate(arg_alloc_size);
  }

  void deallocate(void *ptr, size_t size) const {
    assert(m_allocator.has_value());
    m_allocator->deallocate(ptr);
  }

  void deallocate(const char *arg_label, void *const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) {
    deallocate(arg_alloc_ptr, arg_alloc_size);
  }

  /**\brief Return Name of the MemorySpace */
  static constexpr const char *name() { return "UMPIRE"; }

  static constexpr bool is_host_accessible_space() {
    return Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                           MemorySpace>::accessible;
  }

private:
  static std::optional<umpire::Allocator> m_allocator;
  friend class Kokkos::Impl::SharedAllocationRecord<UmpireSpace<MemorySpace>,
                                                    void>;
};

namespace Kokkos {
namespace Impl {

template <typename T>
concept NotAnonSpace = !std::same_as<T, Kokkos::AnonymousSpace>;

template <NotAnonSpace MemorySpace1, class MemorySpace2, class TagType>
struct MemorySpaceAccess<MemorySpace1, UmpireSpace<MemorySpace2, TagType>> {
  static constexpr bool assignable =
      MemorySpaceAccess<MemorySpace1, MemorySpace2>::assignable;
  static constexpr bool accessible =
      MemorySpaceAccess<MemorySpace1, MemorySpace2>::accessible;
  static constexpr bool deepcopy =
      MemorySpaceAccess<MemorySpace1, MemorySpace2>::deepcopy;
};

template <class MemorySpace1, NotAnonSpace MemorySpace2, class TagType>
struct MemorySpaceAccess<UmpireSpace<MemorySpace1, TagType>, MemorySpace2> {
  static constexpr bool assignable =
      MemorySpaceAccess<MemorySpace1, MemorySpace2>::assignable;
  static constexpr bool accessible =
      MemorySpaceAccess<MemorySpace1, MemorySpace2>::accessible;
  static constexpr bool deepcopy =
      MemorySpaceAccess<MemorySpace1, MemorySpace2>::deepcopy;
};

template <class MemorySpace1, class MemorySpace2, class TagType1,
          class TagType2>
struct MemorySpaceAccess<UmpireSpace<MemorySpace1, TagType1>,
                         UmpireSpace<MemorySpace2, TagType2>> {
  static constexpr bool assignable =
      MemorySpaceAccess<MemorySpace1, MemorySpace2>::assignable;
  static constexpr bool accessible =
      MemorySpaceAccess<MemorySpace1, MemorySpace2>::accessible;
  static constexpr bool deepcopy =
      MemorySpaceAccess<MemorySpace1, MemorySpace2>::deepcopy;
};

template <class MemorySpace1, class MemorySpace2, class TagType,
          class ExecutionSpace>
struct DeepCopy<UmpireSpace<MemorySpace1, TagType>, MemorySpace2,
                ExecutionSpace> {
  inline DeepCopy(void *dst, const void *src, size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace &exec, void *dst, const void *src,
                  size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(exec, dst, src, n);
  }
};

template <class MemorySpace1, class MemorySpace2, class TagType1,
          class TagType2, class ExecutionSpace>
struct DeepCopy<UmpireSpace<MemorySpace1, TagType1>,
                UmpireSpace<MemorySpace2, TagType2>, ExecutionSpace> {
  inline DeepCopy(void *dst, const void *src, size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace &exec, void *dst, const void *src,
                  size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(exec, dst, src, n);
  }
};

template <class MemorySpace1, class MemorySpace2, class TagType,
          class ExecutionSpace>
struct DeepCopy<MemorySpace1, UmpireSpace<MemorySpace2, TagType>,
                ExecutionSpace> {
  inline DeepCopy(void *dst, const void *src, size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(dst, src, n);
  }

  inline DeepCopy(const ExecutionSpace &exec, void *dst, const void *src,
                  size_t n) {
    DeepCopy<MemorySpace1, MemorySpace2, ExecutionSpace>(exec, dst, src, n);
  }
};

} // namespace Impl
} // namespace Kokkos

template <typename MemorySpace, typename TagType>
std::optional<umpire::Allocator> UmpireSpace<MemorySpace, TagType>::m_allocator;

#endif // KOKKOS_UMPIRE_SPACE_HPP
