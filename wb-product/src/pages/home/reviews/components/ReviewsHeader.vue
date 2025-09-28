<template>
  <div class="reviews-header">
    <div class="container">
      <div class="reviews-header-content">
        <button @click="goBack" class="back-btn">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M15 18L9 12L15 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          Назад к товару
        </button>
        <h1 class="reviews-title">Отзывы о товаре</h1>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ReviewsHeader',
  methods: {
    goBack() {
      this.$router.push({ name: 'Product' });
    }
  }
}
</script>

<style scoped>
.reviews-header {
  background: var(--mo-neutral-1000);
  border-bottom: 1px solid var(--mo-neutral-900);
  padding: 16px 0;
}

.reviews-header-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.back-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  background: none;
  border: none;
  color: var(--mo-text-icon-secondary-default);
  cursor: pointer;
  font-size: 14px;
  padding: 8px 0;
  transition: color 0.2s ease;
}

.back-btn:hover {
  color: var(--mo-text-icon-primary-default);
}

.reviews-title {
  font-size: 24px;
  font-weight: 600;
  color: var(--mo-text-icon-primary-default);
  margin: 0;
}

@media (max-width: 768px) {
  .reviews-header-content {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .reviews-title {
    font-size: 20px;
  }
}
</style>
