<template>
  <div class="product-info-section">
    <div class="container">
      <div class="product-card">
        <div class="product-image">
          <img :src="product.images[0]" :alt="product.title" />
        </div>
        <div class="product-details">
          <h2 class="product-title">{{ product.title }}</h2>
          <div class="product-rating-overview">
            <div class="rating-display">
              <div class="stars">
                <span v-for="i in 5" :key="i" class="star" :class="{ 'filled': i <= Math.floor(product.rating) }">★</span>
              </div>
              <span class="rating-value">{{ product.rating }}</span>
            </div>
            <span class="reviews-count">{{ product.reviewsCount }} отзывов</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ProductInfo',
  props: {
    product: {
      type: Object,
      required: true
    }
  }
}
</script>

<style scoped>
.product-info-section {
  padding: 24px 0;
  background: var(--mo-neutral-1000);
  border-bottom: 1px solid var(--mo-neutral-900);
}

.product-card {
  display: flex;
  gap: 16px;
  align-items: center;
}

.product-image {
  width: 80px;
  height: 80px;
  border-radius: 8px;
  overflow: hidden;
  flex-shrink: 0;
}

.product-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.product-details {
  flex: 1;
}

.product-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--mo-text-icon-primary-default);
  margin: 0 0 8px 0;
  line-height: 1.4;
}

.product-rating-overview {
  display: flex;
  align-items: center;
  gap: 12px;
}

.rating-display {
  display: flex;
  align-items: center;
  gap: 8px;
}

.stars {
  display: flex;
  gap: 2px;
}

.star {
  font-size: 16px;
  color: var(--mo-neutral-800);
}

.star.filled {
  color: #ffc107;
}

.rating-value {
  font-size: 16px;
  font-weight: 600;
  color: var(--mo-text-icon-primary-default);
}

.reviews-count {
  font-size: 14px;
  color: var(--mo-text-icon-secondary-default);
}

@media (max-width: 768px) {
  .product-card {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .product-image {
    width: 60px;
    height: 60px;
  }
}
</style>
