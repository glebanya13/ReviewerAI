<template>
  <div class="product-rating" @click="goToReviews">
    <div class="rating-stars">
      <span 
        v-for="star in 5" 
        :key="star"
        class="star"
        :class="{ 
          'star-filled': star <= Math.floor(rating),
          'star-half': star === Math.ceil(rating) && rating % 1 !== 0
        }"
      >
        ★
      </span>
    </div>
    <div class="rating-info">
      <span class="rating-value">{{ rating }}</span>
      <span class="reviews-count">({{ reviewsCount }} отзывов)</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ProductRating',
  props: {
    rating: {
      type: Number,
      required: true,
      default: 0
    },
    reviewsCount: {
      type: Number,
      required: true,
      default: 0
    }
  },
  methods: {
    goToReviews() {
      this.$emit('go-to-reviews')
    }
  }
}
</script>

<style scoped>
.product-rating {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 8px 0;
  transition: opacity 0.2s ease;
}

.product-rating:hover {
  opacity: 0.8;
}

.rating-stars {
  display: flex;
  gap: 2px;
}

.star {
  font-size: 18px;
  color: #ddd;
  transition: color 0.2s ease;
}

.star-filled {
  color: #ffc107;
}

.star-half {
  background: linear-gradient(90deg, #ffc107 50%, #ddd 50%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.rating-info {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 14px;
}

.rating-value {
  font-weight: 600;
  color: var(--wb-dark);
}

.reviews-count {
  color: var(--wb-gray-500);
  text-decoration: underline;
}

.reviews-count:hover {
  color: var(--wb-primary);
}

@media (max-width: 768px) {
  .product-rating {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
  
  .rating-info {
    font-size: 12px;
  }
}
</style>
