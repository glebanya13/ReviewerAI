<template>
  <div class="product-reviews">
    <div class="reviews-header">
      <h3>Отзывы покупателей</h3>
      <router-link to="/reviews" class="btn btn-secondary">
        Все отзывы ({{ totalReviews }})
      </router-link>
    </div>

    <div class="reviews-summary">
      <div class="rating-overview">
        <div class="overall-rating">
          <span class="rating-number">{{ overallRating }}</span>
          <div class="rating-stars">
            <span 
              v-for="star in 5" 
              :key="star"
              class="star"
              :class="{ 'star-filled': star <= Math.floor(overallRating) }"
            >
              ★
            </span>
          </div>
          <span class="rating-text">на основе {{ totalReviews }} отзывов</span>
        </div>
      </div>
    </div>

    <div class="reviews-list">
      <div 
        v-for="review in displayedReviews" 
        :key="review.id"
        class="review-item"
      >
        <div class="review-header">
          <div class="reviewer-info">
            <span class="reviewer-name">{{ review.author }}</span>
            <div class="review-rating">
              <span 
                v-for="star in 5" 
                :key="star"
                class="star"
                :class="{ 'star-filled': star <= review.rating }"
              >
                ★
              </span>
            </div>
          </div>
          <span class="review-date">{{ formatDate(review.date) }}</span>
        </div>
        <div class="review-content">
          <p>{{ review.text }}</p>
        </div>
        <div v-if="review.photos && review.photos.length" class="review-photos">
          <img 
            v-for="photo in review.photos" 
            :key="photo"
            :src="photo" 
            :alt="`Фото от ${review.author}`"
            class="review-photo"
            @click="openPhotoModal(photo)"
          />
        </div>
      </div>
    </div>

    <div v-if="showAll" class="reviews-modal" @click="closeModal">
      <div class="reviews-modal-content" @click.stop>
        <div class="modal-header">
          <h2>Все отзывы</h2>
          <button class="close-btn" @click="closeModal">×</button>
        </div>
        <div class="modal-reviews">
          <div 
            v-for="review in allReviews" 
            :key="review.id"
            class="review-item"
          >
            <div class="review-header">
              <div class="reviewer-info">
                <span class="reviewer-name">{{ review.author }}</span>
                <div class="review-rating">
                  <span 
                    v-for="star in 5" 
                    :key="star"
                    class="star"
                    :class="{ 'star-filled': star <= review.rating }"
                  >
                    ★
                  </span>
                </div>
              </div>
              <span class="review-date">{{ formatDate(review.date) }}</span>
            </div>
            <div class="review-content">
              <p>{{ review.text }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ProductReviews',
  props: {
    overallRating: {
      type: Number,
      required: true,
      default: 0
    },
    totalReviews: {
      type: Number,
      required: true,
      default: 0
    }
  },
  data() {
    return {
      showAll: false,
      allReviews: [
        {
          id: 1,
          author: 'Анна М.',
          rating: 5,
          date: '2024-01-15',
          text: 'Отличный комбинезон! Очень теплый и удобный. Малышу комфортно, качество хорошее.',
          photos: []
        },
        {
          id: 2,
          author: 'Мария К.',
          rating: 4,
          date: '2024-01-10',
          text: 'Хороший комбинезон, но размер немного маловат. Качество материала отличное.',
          photos: []
        },
        {
          id: 3,
          author: 'Елена С.',
          rating: 5,
          date: '2024-01-08',
          text: 'Покупала уже второй раз. Очень довольна качеством и теплотой. Рекомендую!',
          photos: []
        },
        {
          id: 4,
          author: 'Ольга В.',
          rating: 4,
          date: '2024-01-05',
          text: 'Комбинезон хороший, но застежки могли бы быть удобнее. В целом довольна покупкой.',
          photos: []
        },
        {
          id: 5,
          author: 'Ирина П.',
          rating: 5,
          date: '2024-01-03',
          text: 'Отличное качество! Малышу очень нравится, не потеет и не мерзнет. Спасибо!',
          photos: []
        }
      ]
    }
  },
  computed: {
    displayedReviews() {
      return this.allReviews.slice(0, 3)
    }
  },
  methods: {
    closeModal() {
      this.showAll = false
    },
    formatDate(dateString) {
      const date = new Date(dateString)
      return date.toLocaleDateString('ru-RU')
    },
    openPhotoModal(photo) {
    }
  }
}
</script>

<style scoped>
.product-reviews {
  margin-top: 32px;
}

.reviews-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.reviews-header h3 {
  font-size: 20px;
  font-weight: 600;
  color: var(--wb-dark);
}

.reviews-summary {
  margin-bottom: 24px;
  padding: 16px;
  background: var(--wb-gray-100);
  border-radius: 8px;
}

.rating-overview {
  display: flex;
  align-items: center;
  gap: 16px;
}

.overall-rating {
  display: flex;
  align-items: center;
  gap: 8px;
}

.rating-number {
  font-size: 32px;
  font-weight: 700;
  color: var(--wb-primary);
}

.rating-stars {
  display: flex;
  gap: 2px;
}

.star {
  font-size: 16px;
  color: #ddd;
}

.star-filled {
  color: #ffc107;
}

.rating-text {
  font-size: 14px;
  color: var(--wb-gray-600);
}

.reviews-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.review-item {
  padding: 16px;
  border: 1px solid var(--wb-gray-200);
  border-radius: 8px;
  background: var(--wb-white);
}

.review-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.reviewer-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.reviewer-name {
  font-weight: 600;
  color: var(--wb-dark);
}

.review-rating {
  display: flex;
  gap: 2px;
}

.review-date {
  font-size: 14px;
  color: var(--wb-gray-500);
}

.review-content {
  margin-bottom: 12px;
}

.review-content p {
  color: var(--wb-gray-700);
  line-height: 1.5;
}

.review-photos {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.review-photo {
  width: 60px;
  height: 60px;
  object-fit: cover;
  border-radius: 4px;
  cursor: pointer;
  transition: transform 0.2s ease;
}

.review-photo:hover {
  transform: scale(1.1);
}

/* Modal styles */
.reviews-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.reviews-modal-content {
  background: var(--wb-white);
  border-radius: 12px;
  max-width: 800px;
  max-height: 80vh;
  width: 90%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid var(--wb-gray-200);
}

.modal-header h2 {
  font-size: 24px;
  font-weight: 600;
  color: var(--wb-dark);
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--wb-gray-500);
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-btn:hover {
  color: var(--wb-dark);
}

.modal-reviews {
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

@media (max-width: 768px) {
  .reviews-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .rating-overview {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  
  .review-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  
  .reviews-modal-content {
    width: 95%;
    max-height: 90vh;
  }
}
</style>
