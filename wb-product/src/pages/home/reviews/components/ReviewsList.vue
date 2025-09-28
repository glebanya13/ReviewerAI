<template>
  <div class="reviews-list">
    <div v-for="review in reviews" :key="review.id" class="review-item">
      <div class="review-header">
        <div class="reviewer-info">
          <div class="reviewer-avatar">
            <img :src="review.avatar" :alt="review.name" />
          </div>
          <div class="reviewer-details">
            <div class="reviewer-name">{{ review.name }}</div>
            <div class="review-date">{{ formatDate(review.date) }}</div>
          </div>
        </div>
        <div class="review-rating">
          <div class="stars">
            <span v-for="i in 5" :key="i" class="star" :class="{ 'filled': i <= review.rating }">★</span>
          </div>
        </div>
      </div>
      
      <div class="review-content">
        <p>{{ review.text }}</p>
      </div>
      
      <div v-if="review.photos && review.photos.length" class="review-photos">
        <div v-for="photo in review.photos" :key="photo" class="review-photo">
          <img :src="photo" alt="Фото отзыва" />
        </div>
      </div>
      
      <div v-if="review.response" class="review-response">
        <div class="response-header">
          <span class="response-author">{{ review.response.author }}</span>
          <span class="response-date">{{ formatDate(review.response.date) }}</span>
        </div>
        <div class="response-content">
          <p>{{ review.response.text }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ReviewsList',
  props: {
    reviews: {
      type: Array,
      required: true
    }
  },
  methods: {
    formatDate(dateString) {
      const date = new Date(dateString);
      return date.toLocaleDateString('ru-RU', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      });
    }
  }
}
</script>

<style scoped>
.reviews-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.review-item {
  background: var(--mo-neutral-1000);
  border-radius: 8px;
  padding: 20px;
  border: 1px solid var(--mo-neutral-900);
}

.review-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.reviewer-info {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

.reviewer-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  overflow: hidden;
  flex-shrink: 0;
}

.reviewer-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.reviewer-details {
  flex: 1;
}

.reviewer-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--mo-text-icon-primary-default);
  margin-bottom: 4px;
}

.review-date {
  font-size: 12px;
  color: var(--mo-text-icon-secondary-default);
}

.review-rating {
  flex-shrink: 0;
}

.review-content {
  margin-bottom: 16px;
}

.review-content p {
  font-size: 14px;
  line-height: 1.5;
  color: var(--mo-text-icon-primary-default);
  margin: 0;
}

.review-photos {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.review-photo {
  width: 80px;
  height: 80px;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: transform 0.2s ease;
}

.review-photo:hover {
  transform: scale(1.05);
}

.review-photo img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.review-response {
  background: var(--mo-neutral-950);
  border-radius: 8px;
  padding: 16px;
  border-left: 3px solid var(--mo-button-primary-bg-default);
}

.response-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.response-author {
  font-size: 12px;
  font-weight: 600;
  color: var(--mo-text-icon-primary-default);
}

.response-date {
  font-size: 12px;
  color: var(--mo-text-icon-secondary-default);
}

.response-content p {
  font-size: 14px;
  line-height: 1.5;
  color: var(--mo-text-icon-primary-default);
  margin: 0;
}

@media (max-width: 768px) {
  .review-header {
    flex-direction: column;
    gap: 12px;
  }
  
  .reviewer-info {
    width: 100%;
  }
  
  .review-photos {
    gap: 6px;
  }
  
  .review-photo {
    width: 60px;
    height: 60px;
  }
}
</style>
