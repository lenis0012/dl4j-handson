package nl.quintor.handson.dl4j.spring.animals;

public class AnimalInputDTO {
    private int yearsLived;
    private Food eats;
    private Sound sounds;
    private float weight;

    public AnimalInputDTO() {
    }

    public AnimalInputDTO(int yearsLived, Food eats, Sound sounds, float weight) {
        this.yearsLived = yearsLived;
        this.eats = eats;
        this.sounds = sounds;
        this.weight = weight;
    }

    public int getYearsLived() {
        return yearsLived;
    }

    public void setYearsLived(int yearsLived) {
        this.yearsLived = yearsLived;
    }

    public Food getEats() {
        return eats;
    }

    public void setEats(Food eats) {
        this.eats = eats;
    }

    public Sound getSounds() {
        return sounds;
    }

    public void setSounds(Sound sounds) {
        this.sounds = sounds;
    }

    public float getWeight() {
        return weight;
    }

    public void setWeight(float weight) {
        this.weight = weight;
    }
}
