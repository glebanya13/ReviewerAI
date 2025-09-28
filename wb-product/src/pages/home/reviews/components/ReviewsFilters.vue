<template>
  <div class="reviews-filters">
    <div class="filter-group">
      <label>Сортировка:</label>
      <select v-model="sortBy" @change="handleSortChange">
        <option value="newest">Сначала новые</option>
        <option value="oldest">Сначала старые</option>
        <option value="rating-high">Высокий рейтинг</option>
        <option value="rating-low">Низкий рейтинг</option>
      </select>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ReviewsFilters',
  data() {
    return {
      sortBy: 'newest'
    }
  },
  methods: {
    handleSortChange() {
      this.$emit('sort-changed', this.sortBy)
    }
  }
}
</script>

<style scoped>
.reviews-filters {
  margin-bottom: 24px;
  padding: 16px;
  background: var(--mo-neutral-1000);
  border-radius: 8px;
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 12px;
}

.filter-group label {
  font-size: 14px;
  font-weight: 500;
  color: var(--mo-text-icon-primary-default);
}

.filter-group select {
  padding: 8px 12px;
  border: 1px solid var(--mo-neutral-800);
  border-radius: 6px;
  background: var(--mo-neutral-1000);
  color: var(--mo-text-icon-primary-default);
  font-size: 14px;
  cursor: pointer;
}
</style>
