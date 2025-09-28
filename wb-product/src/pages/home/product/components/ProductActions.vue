<template>
  <div class="product-actions">
    <button 
      class="btn btn-primary btn-large" 
      @click="handleAddToCart"
      :disabled="loading"
    >
      <span v-if="loading">Добавление...</span>
      <span v-else>В корзину</span>
    </button>
    
    <button 
      class="btn btn-secondary btn-large" 
      @click="handleAddToFavorites"
      :class="{ 'favorited': isFavorited }"
    >
      <span v-if="isFavorited">♥ В избранном</span>
      <span v-else>♥ В избранное</span>
    </button>
  </div>
</template>

<script>
export default {
  name: 'ProductActions',
  props: {
    productId: {
      type: [String, Number],
      required: true
    }
  },
  data() {
    return {
      loading: false,
      isFavorited: false
    }
  },
  methods: {
    async handleAddToCart() {
      this.loading = true
      try {
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        this.$emit('cart-added', this.productId)
      } catch (error) {
        throw error
      } finally {
        this.loading = false
      }
    },
    
    handleAddToFavorites() {
      this.isFavorited = !this.isFavorited
      
      if (this.isFavorited) {
        this.$emit('favorite-added', this.productId)
      } else {
        this.$emit('favorite-removed', this.productId)
      }
    }
  }
}
</script>

<style scoped>
.product-actions {
  display: flex;
  gap: 16px;
  margin-bottom: 32px;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  text-decoration: none;
  cursor: pointer;
  transition: all 0.2s ease;
  min-height: 48px;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--mo-button-primary-bg-default);
  color: var(--mo-button-primary-content-primary-default);
}

.btn-primary:hover:not(:disabled) {
  background: var(--mo-button-primary-bg-hovered);
  transform: translateY(-1px);
}

.btn-primary:active {
  background: var(--mo-button-primary-bg-pressed);
}

.btn-primary:disabled {
  background: var(--mo-button-primary-bg-disabled);
  color: var(--mo-button-primary-content-primary-disabled);
}

.btn-secondary {
  background: var(--mo-controls-bg-accentLight-default);
  color: var(--mo-controls-text-icon-accent-default);
  border: 1px solid var(--mo-controls-text-icon-accent-default);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--mo-controls-bg-accentLight-hovered);
  color: var(--mo-controls-text-icon-accent-hovered);
}

.btn-secondary:active {
  background: var(--mo-controls-bg-accentLight-pressed);
  color: var(--mo-controls-text-icon-accent-pressed);
}

.btn-secondary.favorited {
  background: var(--mo-button-primary-bg-default);
  color: var(--mo-button-primary-content-primary-default);
  border-color: var(--mo-button-primary-bg-default);
}

.btn-large {
  padding: 16px 32px;
  font-size: 18px;
  min-height: 56px;
}

@media (max-width: 768px) {
  .product-actions {
    flex-direction: column;
  }
  
  .btn {
    width: 100%;
  }
}
</style>
