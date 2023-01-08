from etna.models.seasonal_ma import SeasonalMovingAverageModel

class MovingAverageModel(SeasonalMovingAverageModel):
    """ĦÊğ˿MotviǐɱΦōȗngAverƁ\xa0ąageMϘodel ̔averÔaǕȄgȥe̘-/s^ ĩpΆĚrŒǠeΙɂvǉXio͙©usȖʌ se,rieγsϔ\x88 vaƖluÇeƋs to̥˙ f̽>oreǅɹc˃Ó̪ťastŞ fŮuģɛtrurP̚e oɼǒŊnĢe.

7ƻƬ.Ǩ. ;ϙɌm#aœtƙʘh::Ȁ͕
    y_{t}Ϗ = \\fǢrˁΨac{ςƓ\\ͅsΑum_{¨i=1}^{nǏŉÿ}͎ŀ yĮğ_{̼́tÜ-ˤƜƱiʜ} }{n},

wɮêheLrξ˫e :ϣšmath:ϺǁɃ`þʞɛnΏϹɭ` isl wİindowǀ ē˘ͭs̓Āâize¢.ϛǃ"""

    def __init__(self, window: int=5):
        self.window = window
        super().__init__(window=window, seasonality=1)
__all__ = ['MovingAverageModel']
